#!/usr/bin/env python3
"""Freeze, run and assemble restartable NVT segments; never submit jobs itself."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("segmented_nvt.py", "lmp_segment.in", "submit_segment.slurm")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def endpoints(config):
    total = config["equil_steps"] + config["prod_steps"]
    return list(range(config["segment_steps"], total, config["segment_steps"])) + [total]


def prepare(args):
    output = args.output.resolve()
    if any(c.isspace() for p in (output, args.cell, args.model) for c in str(p)):
        raise ValueError("LAMMPS paths must not contain whitespace")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new, empty output directory")
    config = dict(equil_steps=args.equil_steps, prod_steps=args.prod_steps,
                  segment_steps=args.segment_steps, checkpoint_every=args.checkpoint_every,
                  dump_every=args.dump_every, dt_ps=args.dt_ps, temperature=args.temperature,
                  replicate=args.replicate, seed=299792458, equil_dump_every=1000)
    if any(config[k] <= 0 for k in ("equil_steps", "prod_steps", "segment_steps",
                                    "checkpoint_every", "dump_every", "dt_ps", "temperature")):
        raise ValueError("All timing and temperature settings must be positive")
    if min(args.replicate) < 1:
        raise ValueError("Replication factors must be positive")
    if args.segment_steps <= args.equil_steps:
        raise ValueError("First segment must include equilibration and some production")
    if any(config[k] % args.dump_every for k in
           ("equil_steps", "prod_steps", "segment_steps", "checkpoint_every")):
        raise ValueError("Phase, segment and checkpoint boundaries must align with dump interval")
    header = "\n".join(args.cell.read_text().splitlines()[:20])
    if not re.search(r"^\s*5\s+atoms\s*$", header, re.MULTILINE):
        raise ValueError("Seed must be a minimized five-atom LAMMPS data file")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, source / name)
    shutil.copy2(args.model, output / "model.pb")
    shutil.copy2(args.cell, output / "minimized_cell.data")
    # Resolving the venv's Python symlink would select the base interpreter's
    # directory, which does not contain the installed lmp launcher.
    runtime = Path(sys.executable).absolute().parent
    # Source environment paths are frozen independently of the working checkout.
    if not os.environ.get("DP_LMP_PLUGIN") or not shutil.which("lmp"):
        raise ValueError("Source env.sh first")
    if not (runtime / "lmp").is_file():
        raise ValueError("Use the Python interpreter from the LAMMPS virtual environment")
    config.update(runtime=str(runtime), plugin=os.environ["DP_LMP_PLUGIN"],
                  deepmd_version=version("deepmd-kit"), lammps_version=version("lammps"),
                  runtime_hashes={str(p): sha(p) for p in
                                  (runtime / "lmp", Path(os.environ["DP_LMP_PLUGIN"]))},
                  hashes={str(p.relative_to(output)): sha(p) for p in
                          [output / "model.pb", output / "minimized_cell.data",
                           *(source / name for name in SOURCE_FILES)]},
                  seed_provenance=str(args.cell.resolve()),
                  production_frame_count=args.prod_steps // args.dump_every + 1,
                  clock="absolute MD steps; production time = (step - equil_steps)*dt_ps",
                  domain_status="Single-domain seed; finite-temperature domain stability unverified")
    save(output / "run_manifest.json", config)
    (output / "checkpoints").mkdir()
    (output / "attempts").mkdir()
    print(json.dumps({"output": str(output), "endpoints": endpoints(config),
                      "production_frames": config["production_frame_count"]}, indent=2))


def latest_checkpoint(output):
    files = list((output / "checkpoints").glob("checkpoint.*"))
    files = [p for p in files if p.name.rsplit(".", 1)[1].isdigit() and p.stat().st_size]
    if not files:
        return 0, None
    path = max(files, key=lambda p: int(p.name.rsplit(".", 1)[1]))
    return int(path.name.rsplit(".", 1)[1]), path


def trajectory_ranges(output, config):
    """Latest attempt owns a restart boundary; trim discarded tails of older attempts."""
    records = [json.loads(p.read_text()) for p in sorted((output / "attempts").glob("*/attempt.json"))]
    ranges = []
    for record in records:
        start = record["start_step"]
        ranges = [dict(row, stop=min(row["stop"], start - config["dump_every"]))
                  for row in ranges if row["start"] < start]
        first = max(start, config["equil_steps"])
        if first <= record["until"]:
            ranges.append(dict(path=record["trajectory"], start=first,
                               stop=record["until"]))
    return ranges


def run(args):
    output = args.output.resolve(strict=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((output / "run_manifest.json").read_text())
        for name, expected in config["hashes"].items():
            if sha(output / name) != expected:
                raise RuntimeError(f"Frozen input changed: {name}")
        for name, expected in config["runtime_hashes"].items():
            if sha(name) != expected:
                raise RuntimeError(f"Runtime executable/plugin changed: {name}")
        if version("deepmd-kit") != config["deepmd_version"] or version("lammps") != config["lammps_version"]:
            raise RuntimeError("Runtime version changed")
        until = args.until
        if until not in endpoints(config):
            raise ValueError("--until must be a planned segment endpoint")
        step, restart = latest_checkpoint(output)
        progress = output / "progress.json"
        completed = json.loads(progress.read_text())["completed_step"] if progress.exists() else 0
        previous = max([0] + [x for x in endpoints(config) if x < until])
        if completed < previous:
            raise RuntimeError("Previous segment did not complete; resume that segment first")
        if completed >= until:
            print(f"Already completed through {completed}; nothing to run")
            return
        if step > until:
            raise RuntimeError("Checkpoint is beyond requested endpoint")
        attempt = output / "attempts" / f"{len(list((output/'attempts').iterdir())) + 1:04d}"
        attempt.mkdir()
        variables = dict(fresh=int(restart is None), restartfile=restart or "unused",
                         cellfile=output / "minimized_cell.data", model=output / "model.pb",
                         nx=config["replicate"][0], ny=config["replicate"][1], nz=config["replicate"][2],
                         dt=config["dt_ps"], temperature=config["temperature"], seed=config["seed"],
                         equil_steps=config["equil_steps"], dump_every=config["dump_every"],
                         equil_dump_every=config["equil_dump_every"], until=until,
                         checkpoint_every=config["checkpoint_every"],
                         checkpoint_dir=output / "checkpoints", outdir=attempt)
        command = [str(Path(config["runtime"]) / "lmp"), "-in", str(output / "source/lmp_segment.in"),
                   "-log", str(attempt / "lammps.log")]
        for key, value in variables.items():
            command.extend(["-var", key, str(value)])
        record = dict(start_step=step, until=until, job_id=os.environ.get("SLURM_JOB_ID"),
                      trajectory=str((attempt / "production.lammpstrj").relative_to(output)),
                      command=command, status="running")
        save(attempt / "attempt.json", record)
        print(f"NVT {step} -> {until}; progress in {attempt/'screen.txt'}", flush=True)
        with (attempt / "screen.txt").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        record["returncode"] = result.returncode
        record["status"] = "complete" if result.returncode == 0 else "failed"
        save(attempt / "attempt.json", record)
        if result.returncode:
            raise RuntimeError(f"LAMMPS failed; inspect {attempt/'screen.txt'}")
        if f"SEGMENT_COMPLETE {until}" not in (attempt / "screen.txt").read_text():
            raise RuntimeError("Missing segment completion marker")
        if not (output / "checkpoints" / f"checkpoint.{until}").is_file():
            raise RuntimeError("Missing segment end restart")
        save(progress, dict(completed_step=until, total_steps=config["equil_steps"]+config["prod_steps"]))
        save(output / "trajectory_index.json", dict(dt_ps=config["dt_ps"],
             production_origin_step=config["equil_steps"], dump_every=config["dump_every"],
             completed_step=until, ranges=trajectory_ranges(output, config)))
        print(f"Completed through {until}; trajectory index updated", flush=True)


def assemble(args):
    """Stream a completed run to one dump, removing boundary/rollback overlaps."""
    output = args.output.resolve(strict=True)
    index = json.loads((output / "trajectory_index.json").read_text())
    config = json.loads((output / "run_manifest.json").read_text())
    if index["completed_step"] != config["equil_steps"] + config["prod_steps"]:
        raise RuntimeError("Run is not complete")
    expected = config["equil_steps"]
    # Exclusive creation preserves existing data. On error the destination is
    # incomplete; inspect/remove it explicitly before retrying.
    with args.destination.open("x") as dest:
        for row in index["ranges"]:
            with (output / row["path"]).open() as source:
                while True:
                    header = source.readline()
                    if not header:
                        break
                    if header != "ITEM: TIMESTEP\n":
                        raise ValueError("Malformed trajectory header")
                    step_line = source.readline()
                    step = int(step_line)
                    if step > row["stop"]:
                        break  # Discard a rolled-back tail, even if truncated.
                    number_header, count_line = source.readline(), source.readline()
                    if number_header != "ITEM: NUMBER OF ATOMS\n":
                        raise ValueError("Malformed atom-count header")
                    count = int(count_line)
                    selected = row["start"] <= step <= row["stop"]
                    if selected:
                        if step != expected:
                            raise ValueError(f"Trajectory gap/overlap at {step}, expected {expected}")
                        dest.write(header + step_line + number_header + count_line)
                    for _ in range(5 + count):
                        line = source.readline()
                        if not line:
                            raise ValueError("Incomplete trajectory frame")
                        if selected:
                            dest.write(line)
                    if selected:
                        expected += config["dump_every"]
    if expected != config["equil_steps"] + config["prod_steps"] + config["dump_every"]:
        raise ValueError("Missing final production frame")
    print(f"Assembled {config['production_frame_count']} frames in {args.destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--model", type=Path, required=True)
    prep.add_argument("--cell", type=Path, required=True)
    prep.add_argument("--equil-steps", type=int, default=10000)
    prep.add_argument("--prod-steps", type=int, default=405000)
    prep.add_argument("--segment-steps", type=int, default=50000)
    prep.add_argument("--checkpoint-every", type=int, default=1000)
    prep.add_argument("--dump-every", type=int, default=5)
    prep.add_argument("--dt-ps", type=float, default=0.001)
    prep.add_argument("--temperature", type=float, default=300.0)
    prep.add_argument("--replicate", type=int, nargs=3, default=[10, 10, 50])
    execution = commands.add_parser("run")
    execution.add_argument("--output", type=Path, required=True)
    execution.add_argument("--until", type=int, required=True)
    merge = commands.add_parser("assemble")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "assemble": assemble}[args.action](args)


if __name__ == "__main__":
    main()
