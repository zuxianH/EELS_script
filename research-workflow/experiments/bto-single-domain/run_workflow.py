#!/usr/bin/env python3
"""Prepare a +x single-domain seed by minimization, then run unrestricted NVT."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from trajectory_tools import analyze, read_dump, ti_offsets

ROOT = Path(__file__).resolve().parent


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "minimize", "nvt"), default="all")
    parser.add_argument("--output", type=Path, required=True, help="new, empty run directory")
    parser.add_argument("--model", type=Path, default=ROOT / "BTO_pbesol_final_compressed.pb")
    parser.add_argument("--cell", type=Path, help="minimized five-atom cell for --stage nvt")
    parser.add_argument("--relax-box", type=int, choices=(0, 1), default=1)
    parser.add_argument("--replicate", nargs=3, type=positive_int, default=(10, 10, 50), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--dt-ps", type=float, default=0.001)
    parser.add_argument("--equil-steps", type=positive_int, default=10000)
    parser.add_argument("--prod-steps", type=positive_int, default=10000)
    parser.add_argument("--dump-every", type=positive_int, default=5)
    parser.add_argument("--equil-dump-every", type=positive_int, default=1000)
    parser.add_argument("--seed", type=positive_int, default=299792458)
    parser.add_argument("--ftol", type=float, default=1e-6)
    parser.add_argument("--min-iterations", type=positive_int, default=5000)
    parser.add_argument("--min-evaluations", type=positive_int, default=50000)
    parser.add_argument("--min-cycles", type=positive_int, default=3)
    args = parser.parse_args()
    if min(args.temperature, args.dt_ps, args.ftol) <= 0:
        parser.error("temperature, dt-ps, and ftol must be positive")
    if args.prod_steps % args.dump_every:
        parser.error("prod-steps must be divisible by dump-every for uniform sampling")
    if not 1 <= args.seed <= 900000000:
        parser.error("seed must be between 1 and 900000000")
    if args.stage == "nvt" and args.cell is None:
        parser.error("--stage nvt requires --cell")
    if args.stage == "nvt":
        with args.cell.open() as stream:
            header = "".join(stream.readline() for _ in range(20))
        atom_count = re.search(r"^\s*(\d+)\s+atoms\s*$", header, re.MULTILINE)
        if atom_count is None or int(atom_count.group(1)) != 5:
            parser.error("--cell must be a five-atom minimized LAMMPS data file")
    if not os.environ.get("DP_LMP_PLUGIN") or not shutil.which("lmp"):
        parser.error("source env.sh before running this script")
    model = args.model.resolve(strict=True)
    output = args.output.resolve()
    for path in (ROOT, model, output, args.cell):
        if path is not None and any(c.isspace() for c in str(path)):
            parser.error("LAMMPS input paths must not contain whitespace")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from deepmd.infer import DeepPot
    dp = DeepPot(str(model))
    if set(dp.get_type_map()) != {"Ba", "Ti", "O"}:
        raise ValueError(f"Unexpected potential type map: {dp.get_type_map()}")
    if dp.get_dim_fparam() or dp.get_dim_aparam():
        raise ValueError("Potential requires extra parameters not configured by this workflow")
    metadata = vars(args).copy()
    metadata.update(model=str(model), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    model_type_map=dp.get_type_map(), model_cutoff_A=dp.get_rcut(),
                    deepmd_version=version("deepmd-kit"), lammps_version=version("lammps"),
                    python=sys.version, frame_interval_ps=args.dt_ps*args.dump_every,
                    intended_atoms=5*int(np.prod(args.replicate)),
                    preparation="five-atom +x tetragonal branch; transverse symmetry only during minimization",
                    MD="unrestricted NVT; single-domain order must be checked after equilibration")
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    del dp

    def run_input(name, folder, variables):
        folder.mkdir()
        shutil.copy2(ROOT / name, folder / name)
        cmd = [shutil.which("lmp"), "-in", str(ROOT/name), "-log", str(folder/"lammps.log"),
               "-var", "model", str(model), "-var", "outdir", str(folder)]
        for key, value in variables.items():
            cmd.extend(["-var", key, str(value)])
        (folder / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        print(f"Running {name}; progress: {folder/'screen.txt'}", flush=True)
        with (folder / "screen.txt").open("w") as handle:
            subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)

    if args.stage != "nvt":
        folder = output / "minimization"
        run_input("lmp_minimize.in", folder, dict(relax_box=args.relax_box, ftol=args.ftol,
                  maxiter=args.min_iterations, maxeval=args.min_evaluations, cycles=args.min_cycles))
        frame = next(read_dump(folder / "minimized_cell.lammpstrj"))
        _, offsets, _ = ti_offsets(frame)
        if len(frame.ids) != 5 or offsets[0, 0] < 0.01 or np.linalg.norm(offsets[0, 1:]) > 0.001:
            raise RuntimeError(f"Minimized seed is not +x polarized: offsets={offsets.tolist()}. Inspect minimization.")
        summary = analyze(folder / "minimized_cell.lammpstrj", folder / "domain_check")
        print("Minimized Ti-to-six-O-center vector (A):", offsets[0].tolist(), flush=True)
        print("Minimized cell lengths (A):", frame.lengths.tolist(), flush=True)
        # Capture the actual physical final forces, not the projected forces.
        screen = (folder / "screen.txt").read_text()
        force_line = next(line for line in reversed(screen.splitlines())
                          if line.startswith("Unprojected maximum atomic force component"))
        final_force = float(force_line.rsplit(":", 1)[1])
        if final_force > max(1e-5, 10*args.ftol):
            raise RuntimeError(f"Minimization insufficiently converged: physical fmax={final_force:g} eV/A")
        cell = folder / "minimized_cell.data"
    else:
        cell = args.cell.resolve(strict=True)
    if args.stage != "minimize":
        folder = output / "nvt"
        run_input("lmp_nvt.in", folder, dict(cellfile=cell, nx=args.replicate[0], ny=args.replicate[1],
                  nz=args.replicate[2], temperature=args.temperature, seed=args.seed, dt=args.dt_ps,
                  equil_steps=args.equil_steps, prod_steps=args.prod_steps, dump_every=args.dump_every,
                  equil_dump_every=args.equil_dump_every))
        summary = analyze(folder / "production.lammpstrj", folder / "domain_check")
        print(json.dumps(summary, indent=2), flush=True)
        print("Inspect nvt/domain_check/ti_time_averaged.csv spatially before using spectra.", flush=True)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
