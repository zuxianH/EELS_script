"""Command-line runtime copied into each generated run folder.

Only calculation commands import Torch, pyms, or torched_tacaw.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from filelock import FileLock

from .config import effective_parameters, validate_settings, validate_parameters
from .geometry import coordinates
from .storage import atomic_json, atomic_yaml, read_yaml, convert_run, require_complete
from .trajectories import (compare_frames, convert, needs_conversion, supported_cell,
                           validate_beads, write_structure)

CONFIG_NAME = "scan_gpu_config.yaml"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def lifecycle(folder: Path, state: str, **extra):
    atomic_json(folder/"lifecycle.json", dict(state=state, updated=timestamp(), **extra))


def check_environment(backend: str, *, check_device=True) -> dict:
    packages = {}
    for name in ("numpy", "ase", "yaml", "zarr", "filelock", "scipy", "torch", "pyms", "torched_tacaw"):
        try:
            module = importlib.import_module(name)
            package_version = getattr(module, "__version__", None)
            if package_version is None:
                try:
                    package_version = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    package_version = "unknown"
            packages[name] = {"version": str(package_version),
                              "path": str(getattr(module, "__file__", ""))}
        except Exception as exc:
            raise RuntimeError(f"Cannot import {name} using {sys.executable}: {exc}. Configure the simulation Python, launcher and PYTHONPATH.") from exc
    import torch
    detail = {"python": sys.executable, "packages": packages, "backend": backend,
              "cuda_version": torch.version.cuda, "hip_version": getattr(torch.version, "hip", None),
              "visible_gpus": torch.cuda.device_count()}
    from .resources import node_memory
    detail["node_memory"] = node_memory()
    detail["gpu_devices"] = [
        {"index": index, "name": torch.cuda.get_device_properties(index).name,
         "total_bytes": torch.cuda.get_device_properties(index).total_memory}
        for index in range(torch.cuda.device_count())] if backend != "cpu" else []
    if check_device and backend != "cpu":
        if backend == "rocm" and not getattr(torch.version, "hip", None):
            raise RuntimeError("ROCm profile requires a HIP/ROCm Torch build.")
        if backend == "cuda" and (not torch.version.cuda or getattr(torch.version, "hip", None)):
            raise RuntimeError("CUDA profile requires an NVIDIA CUDA Torch build.")
        if not torch.cuda.is_available():
            raise RuntimeError("No usable GPU. Check the allocation and container GPU flags.")
        # A real allocation check, beyond device enumeration.
        test = torch.ones(1, device="cuda:0")
        torch.cuda.synchronize()
        del test
    print(json.dumps(detail, indent=2), flush=True)
    return detail


def chunk_starts(p: dict, length: int) -> list[int]:
    span = p["trajectory_chunks_size"] * p["trajectory_chunks_step"]
    hop = int(span / p["trajectory_chunks_overlap"])
    starts = list(range(p["trajectory_chunks_skip_init"], length-span, hop))
    if p["trajectory_chunks_nof"] is not None:
        starts = starts[:p["trajectory_chunks_nof"]]
    if not starts:
        raise ValueError(f"No TACAW chunks fit in {length} prepared frames. Reduce skipped frames/chunk size/step. The current engine requires one extra frame beyond the chunk span.")
    return starts


def finalize_roi(config):
    """Complete center/None crop metadata absent in the supplied engine."""
    k = config["simulation"]["kspace"]
    full = np.asarray(k["shape_full"], dtype=int)
    limited = np.asarray(k["shape_bandwidth_limited"], dtype=int)
    bandwidth_start = np.ceil((full-limited)/2).astype(int)
    if k["ROI_mode"] in (None, "center"):
        shape = np.asarray(k["ROI_shape"], dtype=int)
        start = bandwidth_start if k["ROI_mode"] is None else full//2-shape//2
        k["ROI_min_indices"] = start.tolist()
        k["ROI_max_indices"] = (start+shape).tolist()
    start, stop = np.array(k["ROI_min_indices"]), np.array(k["ROI_max_indices"])
    if np.any(start < bandwidth_start) or np.any(stop > bandwidth_start+limited):
        raise ValueError("Reciprocal-space ROI extends beyond the propagated bandwidth-limited grid. Reduce the ROI or increase the full k-space grid.")


def prepare(folder: Path):
    from ase.io.trajectory import Trajectory
    from ase.io import read
    from torched_tacaw import Config
    settings = read_yaml(folder/"studio.yaml")
    validate_settings(settings, require_files=True)
    config_path = folder/CONFIG_NAME
    if config_path.exists() or (folder/settings["intensities_name"]).exists():
        raise FileExistsError("This run already contains a configuration or result store. Use Resume or create a new run.")
    lifecycle(folder, "preparing")
    options = settings["inputs"]["conversion"]
    paths, provenance = [], []
    # Reject partial initialization retries; a fresh folder makes ownership clear.
    for bead, source in enumerate(settings["inputs"]["files"]):
        source_path = Path(source).expanduser().resolve()
        if needs_conversion(str(source_path), options):
            target = folder/"trajectories"/f"bead_{bead:03}.traj"
            count = convert(str(source_path), target, options)
        else:
            target = source_path
            with Trajectory(target) as r:
                count = len(r)
        paths.append(str(target))
        provenance.append({"source": str(source_path), "prepared": str(target),
                           "frames": count, "source_bytes": source_path.stat().st_size,
                           "source_mtime_ns": source_path.stat().st_mtime_ns,
                           "prepared_bytes": target.stat().st_size,
                           "prepared_mtime_ns": target.stat().st_mtime_ns,
                           "selection": options})
    summary = validate_beads(paths)
    p = effective_parameters(settings)
    p["trajectory_timestep_fs"] *= options["stride"]
    validate_parameters(p)
    starts = chunk_starts(p, summary["frames"])
    with Trajectory(paths[0]) as reader:
        first = reader[0]
        supported_cell(first)
        unitcell = first.cell.array
    if p.get("sample_structure_file"):
        structure = read(Path(p["sample_structure_file"]).expanduser())
        supported_cell(structure)
        unitcell = structure.cell.array
    else:
        p["sample_structure_file"] = str(folder/"structure.extxyz")
    write_structure(paths[0], folder/"structure.extxyz")
    p.update(datafolder=str(folder)+"/", config_file=str(config_path),
             intensities_zarray=str(folder/settings["intensities_name"]),
             trajectory_files=paths, dump_to_yaml=False)
    frequencies = np.fft.fftfreq(p["trajectory_chunks_size"],
                                p["trajectory_timestep_fs"]*p["trajectory_chunks_step"]/1000.)
    lo, hi = p["frequency_THz_ROI"]
    if not np.any((frequencies >= lo) & (frequencies <= hi)):
        raise ValueError("Frequency ROI contains no FFT bins for this chunk size and timestep.")
    try:
        config = Config(**p)
    except IndexError as exc:
        raise ValueError("Reciprocal-space ROI contains no grid bins. Adjust its bounds, beam energy, or grid size.") from exc
    finalize_roi(config)
    statuses = config["computation_batches"]["status_list"]
    if not statuses:
        raise ValueError("TACAW generated no computation batches.")
    if config["simulation"]["frequency_THz"]["ROI_len"] < 1:
        raise ValueError("Frequency ROI contains no FFT bins.")
    # Validate beads on every selected time index; single trajectories are also
    # checked for ordering/cell consistency without loading all frames at once.
    selected = sorted({start + i*p["trajectory_chunks_step"]
                       for start in starts for i in range(p["trajectory_chunks_size"])})
    with ExitStack() as stack:
        readers = [stack.enter_context(Trajectory(path)) for path in paths]
        numbers = readers[0][0].numbers.copy()
        cell = readers[0][0].cell.array
        for index in selected:
            atoms = [r[index] for r in readers]
            compare_frames(atoms, index)
            if not np.array_equal(atoms[0].numbers, numbers) or not np.allclose(atoms[0].cell.array, cell):
                raise ValueError(f"Atom ordering or cell varies at selected frame {index}; this engine requires a fixed simulation cell.")
    if summary["beads"] > 1 and not summary["time_metadata_available"]:
        print("Bead time metadata absent: synchronization is the user's responsibility.", flush=True)
    atomic_yaml(folder/"provenance.yaml", {"inputs": provenance, "summary": summary,
                                          "effective_timestep_fs": p["trajectory_timestep_fs"],
                                          "selected_frame_count": len(selected)})
    grid = coordinates(p, unitcell)
    np.save(folder/"scan_coordinates.npy", grid)
    np.savetxt(folder/"scan_coordinates.csv", np.column_stack([
        np.indices(grid.shape[:2]).reshape(2, -1).T, grid.reshape(-1, 2)]),
        delimiter=",", header="probe_i,probe_j,x_A,y_A", comments="")
    # create_zarr_array(overwrite=True) is allowed only in a fresh run, once.
    config.dump_to_yaml(str(config_path))
    config.create_zarr_array()
    lifecycle(folder, "prepared", batches=len(statuses), frames=summary["frames"])
    print(f"Prepared {len(statuses)} batches in {folder}", flush=True)


def resume(folder: Path):
    from torched_tacaw import Config
    config_path = folder/CONFIG_NAME
    if not config_path.is_file():
        raise ValueError("Resume requires a successfully prepared TACAW configuration.")
    with FileLock(str(config_path)+".lock"):
        config = Config.load_from_yaml(str(config_path))
        from .storage import inspect_array
        array = inspect_array(Path(config["storage"]["intensities_zarray"]))
        expected = [1, config["simulation"]["frequency_THz"]["ROI_len"],
                    *config["beam"]["scanning"]["shape"], *config["simulation"]["kspace"]["ROI_shape"]]
        if array["shape"] != expected:
            raise ValueError("Result store does not match the saved configuration.")
        provenance = read_yaml(folder/"provenance.yaml")
        for item in provenance["inputs"]:
            source = Path(item["source"])
            if not source.is_file() or source.stat().st_size != item["source_bytes"] or source.stat().st_mtime_ns != item["source_mtime_ns"]:
                raise ValueError(f"Source trajectory changed since preparation: {source}")
            prepared = Path(item["prepared"])
            if not prepared.is_file() or prepared.stat().st_size != item.get("prepared_bytes", prepared.stat().st_size) or prepared.stat().st_mtime_ns != item.get("prepared_mtime_ns", prepared.stat().st_mtime_ns):
                raise ValueError(f"Prepared trajectory changed since preparation: {prepared}")
        config["computation_batches"]["status_list"] = [
            0 if value == 1 else value for value in config["computation_batches"]["status_list"]]
        config.dump_to_yaml(str(config_path))
    lifecycle(folder, "prepared", resumed=True)
    print("Reset unfinished batches once. Existing result store retained.", flush=True)


def worker(folder: Path, backend: str, binding: str):
    import torch
    from torched_tacaw import Master
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    if backend == "cpu":
        device = "cpu"
    else:
        check_environment(backend)
        if binding == "local-rank":
            device = f"cuda:{local_rank}"
        else:
            if "SLURM_PROCID" in os.environ and binding == "scheduler" and torch.cuda.device_count() != 1:
                raise ValueError("Scheduler binding must expose one GPU per task. Select local-rank or rocr-local-rank for an allocation exposing the whole node.")
            device = "cuda:0"
        if "SLURM_PROCID" not in os.environ:
            requested = read_yaml(folder/"studio.yaml")["parameters"].get("device", "cpu")
            if requested.startswith("cuda:"):
                device = requested
        if int(device.split(":")[1]) >= torch.cuda.device_count():
            raise ValueError(f"Requested device {device} is not visible.")
    identity = f"{os.environ.get('SLURM_JOB_ID', 'local')}_node{os.environ.get('SLURM_NODEID', '0')}_rank{rank}"
    logfile = folder/"logs"/f"worker_{identity}_{timestamp().replace(':', '-')}.log"
    logfile.parent.mkdir(exist_ok=True)
    logging.basicConfig(filename=logfile, level=logging.INFO, format="%(asctime)s | %(message)s")
    print(f"Worker {rank}: {device}, log {logfile}", flush=True)
    master = Master(str(folder/CONFIG_NAME), logger=logging.getLogger("tacaw"), device=device)
    master.run()


def finish(folder: Path):
    config = require_complete(folder/CONFIG_NAME)
    lifecycle(folder, "calculation_complete", batches=len(config["computation_batches"]["status_list"]))
    if read_yaml(folder/"studio.yaml").get("auto_npy"):
        lifecycle(folder, "converting")
        output = convert_run(folder)
        lifecycle(folder, "complete", npy=str(output))
    else:
        lifecycle(folder, "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare", "resume", "worker", "finish", "convert"))
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm"), default="cpu")
    parser.add_argument("--binding", choices=("scheduler", "local-rank", "rocr-local-rank"), default="scheduler")
    args = parser.parse_args()
    folder = args.folder.expanduser().resolve()
    try:
        if args.action == "check":
            info = check_environment(args.backend)
            atomic_json(folder/"environment.json", info)
        elif args.action == "prepare":
            prepare(folder)
        elif args.action == "resume":
            resume(folder)
        elif args.action == "worker":
            worker(folder, args.backend, args.binding)
        elif args.action == "finish":
            finish(folder)
        else:
            lifecycle(folder, "converting")
            output = convert_run(folder)
            lifecycle(folder, "complete", npy=str(output))
    except Exception as exc:
        # Worker processes never overwrite each other's aggregate lifecycle;
        # their errors remain in logs and the enclosing launch fails.
        if args.action != "worker":
            lifecycle(folder, "failed", action=args.action, error=str(exc))
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
