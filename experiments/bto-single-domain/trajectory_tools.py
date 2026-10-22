#!/usr/bin/env python3
"""Stream orthorhombic LAMMPS dumps: check TiO6 offsets or convert for spectra."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Frame:
    step: int
    ids: np.ndarray
    types: np.ndarray
    positions: np.ndarray
    lower: np.ndarray
    lengths: np.ndarray
    velocities: np.ndarray | None = None


def read_dump(path):
    """Read one frame at a time; require periodic orthorhombic x,y,z dumps."""
    with Path(path).open() as stream:
        while True:
            header = stream.readline()
            if not header:
                return
            if header.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Expected TIMESTEP, found {header!r}")
            step = int(stream.readline())
            if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Missing NUMBER OF ATOMS")
            count = int(stream.readline())
            bounds_header = stream.readline().split()
            if bounds_header != ["ITEM:", "BOX", "BOUNDS", "pp", "pp", "pp"]:
                raise ValueError("Only fully periodic orthorhombic dumps are supported")
            bounds = np.asarray([stream.readline().split() for _ in range(3)], float)
            lengths = bounds[:, 1] - bounds[:, 0]
            if np.any(lengths <= 0):
                raise ValueError("Nonpositive box length")
            atom_header = stream.readline().split()
            if atom_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("Missing ATOMS header")
            columns = {name: i for i, name in enumerate(atom_header[2:])}
            if not {"id", "type", "x", "y", "z"} <= columns.keys():
                raise ValueError("Dump must contain id type x y z")
            ids = np.empty(count, dtype=np.int64)
            types = np.empty(count, dtype=np.int64)
            positions = np.empty((count, 3))
            velocities = np.empty_like(positions) if {"vx", "vy", "vz"} <= columns.keys() else None
            for i in range(count):
                row = stream.readline().split()
                if len(row) != len(columns):
                    raise ValueError(f"Truncated/invalid atom row at step {step}")
                ids[i] = int(row[columns["id"]])
                types[i] = int(row[columns["type"]])
                positions[i] = [float(row[columns[k]]) for k in ("x", "y", "z")]
                if velocities is not None:
                    velocities[i] = [float(row[columns[k]]) for k in ("vx", "vy", "vz")]
            order = np.argsort(ids)
            if np.unique(ids).size != count:
                raise ValueError("Duplicate atom IDs")
            positions = np.mod(positions - bounds[:, 0], lengths) + bounds[:, 0]
            yield Frame(step, ids[order], types[order], positions[order], bounds[:, 0], lengths,
                        None if velocities is None else velocities[order])


def ti_offsets(frame):
    """Ti minus centroid of its six nearest periodic oxygen images, in A.

    Explicit image tiling handles primitive cells, whose six oxygen neighbors
    include multiple periodic images of the same three oxygen atom IDs.
    """
    ti_mask = frame.types == 2
    oxygen = (frame.positions[frame.types == 3] - frame.lower) % frame.lengths
    titanium = (frame.positions[ti_mask] - frame.lower) % frame.lengths
    if len(oxygen) == 0 or len(titanium) == 0:
        raise ValueError("Expected type 2 = Ti and type 3 = O")
    if len(oxygen) < 6 or min(frame.lengths) < 6.0:
        shifts = np.asarray(list(product((-1, 0, 1), repeat=3))) * frame.lengths
        images = (oxygen[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        distances, neighbors = cKDTree(images).query(titanium, k=6)
        relative = images[neighbors] - titanium[:, None, :]
    else:
        distances, neighbors = cKDTree(oxygen, boxsize=frame.lengths).query(titanium, k=6)
        relative = oxygen[neighbors] - titanium[:, None, :]
        relative -= np.rint(relative / frame.lengths) * frame.lengths
    offsets = -relative.mean(axis=1)
    if not np.isfinite(offsets).all():
        raise ValueError("Nonfinite Ti-to-oxygen-center offsets")
    return frame.ids[ti_mask], offsets, distances[:, -1]


def vector_summary(offsets):
    mean = offsets.mean(axis=0)
    local = np.linalg.norm(offsets, axis=1).mean()
    coherent = float(np.linalg.norm(mean))
    return dict(mean_dx_A=float(mean[0]), mean_dy_A=float(mean[1]), mean_dz_A=float(mean[2]),
                mean_local_magnitude_A=float(local), magnitude_of_mean_A=coherent,
                coherence_ratio=coherent / local if local > 1e-15 else 0.0,
                positive_x_fraction=float(np.mean(offsets[:, 0] > 0)),
                positive_x_dominant_fraction=float(np.mean(
                    offsets[:, 0] > np.max(np.abs(offsets[:, 1:]), axis=1))))


def analyze(path, outdir, *, start_step=0, stride=1):
    """Write per-frame summaries and per-Ti time averages; no automatic domain claim."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    sum_offsets = reference_ids = first_positions = last_frame = None
    initial_step = None
    with (outdir / "domain_frames.csv").open("w", newline="") as handle:
        writer = None
        eligible = 0
        for frame in read_dump(path):
            if frame.step < start_step:
                continue
            eligible += 1
            if (eligible - 1) % stride:
                continue
            ids, offsets, radii = ti_offsets(frame)
            if float(radii.max()) > 3.0:
                raise ValueError(f"Step {frame.step}: sixth O farther than 3 A; check cage topology")
            if reference_ids is None:
                reference_ids = ids.copy()
                sum_offsets = np.zeros_like(offsets)
                first_positions = frame.positions[frame.types == 2].copy()
                initial_step = frame.step
            elif not np.array_equal(ids, reference_ids):
                raise ValueError("Ti atom IDs changed between frames")
            sum_offsets += offsets
            row = dict(step=frame.step, **vector_summary(offsets), max_sixth_O_distance_A=float(radii.max()))
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            frame_count += 1
            last_frame = frame
    if not frame_count:
        raise ValueError("No selected frames")
    averaged = sum_offsets / frame_count
    with (outdir / "ti_time_averaged.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "initial_x_A", "initial_y_A", "initial_z_A", "mean_dx_A", "mean_dy_A", "mean_dz_A"])
        for atom_id, pos, offset in zip(reference_ids, first_positions, averaged):
            writer.writerow([int(atom_id), *pos, *offset])
    summary = dict(frames=frame_count, first_step=initial_step, last_step=last_frame.step,
                   n_titanium=len(reference_ids), time_averaged_local_vectors=vector_summary(averaged),
                   note="Inspect spatial Ti averages for domains; thermal sign changes alone are not domain walls.")
    (outdir / "domain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def convert(path, output, dt_ps):
    """Write ASE .traj or extended XYZ in stable atom-ID order, including cell/time."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if output.suffix not in (".xyz", ".traj"):
        raise ValueError("Choose .xyz or .traj output")
    output.parent.mkdir(parents=True, exist_ok=True)
    symbols = np.asarray(["", "Ba", "Ti", "O"])
    previous_ids = previous_types = previous_lengths = None
    previous_step = step_gap = None
    count = 0
    if output.suffix == ".traj":
        from ase import Atoms
        from ase.io.trajectory import Trajectory
        from ase.units import fs
        destination = Trajectory(str(output), "w")
    else:
        destination = output.open("w")
    try:
        for frame in read_dump(path):
            if np.any((frame.types < 1) | (frame.types > 3)):
                raise ValueError("Expected types 1,2,3 = Ba,Ti,O")
            if previous_ids is not None:
                if not np.array_equal(frame.ids, previous_ids) or not np.array_equal(frame.types, previous_types):
                    raise ValueError("Atom identities changed")
                if not np.allclose(frame.lengths, previous_lengths, rtol=0, atol=1e-8):
                    raise ValueError("Box changed; expected an NVT production trajectory")
                gap = frame.step - previous_step
                if gap <= 0 or (step_gap is not None and gap != step_gap):
                    raise ValueError("Frame steps are not uniformly increasing")
                step_gap = gap
            element = symbols[frame.types]
            time_ps = frame.step * dt_ps
            # ASE/extended XYZ place the cell origin at zero. Box relaxation
            # can move LAMMPS lower bounds, so translate the entire frame.
            exported_positions = frame.positions - frame.lower
            if output.suffix == ".traj":
                atoms = Atoms(element.tolist(), positions=exported_positions, cell=frame.lengths, pbc=True)
                atoms.info.update(step=frame.step, time_ps=time_ps, timestep_ps=dt_ps)
                # ASE's native .traj format does not serialize arbitrary arrays:
                # store the sorted LAMMPS IDs as frame metadata instead.
                atoms.info["lammps_ids"] = frame.ids.tolist()
                if frame.velocities is not None:
                    atoms.set_velocities(frame.velocities / (1000.0 * fs))
                destination.write(atoms)
            else:
                lattice = " ".join(f"{x:.15g}" for x in np.diag(frame.lengths).ravel())
                destination.write(f'{len(frame.ids)}\nLattice="{lattice}" Properties=species:S:1:pos:R:3:id:I:1 '
                                  f'pbc="T T T" step={frame.step} time_ps={time_ps:.15g} timestep_ps={dt_ps:.15g}\n')
                for atom_id, symbol, pos in zip(frame.ids, element, exported_positions):
                    destination.write(f"{symbol} {pos[0]:.15g} {pos[1]:.15g} {pos[2]:.15g} {atom_id}\n")
            previous_ids, previous_types = frame.ids, frame.types
            previous_lengths, previous_step = frame.lengths, frame.step
            count += 1
    finally:
        destination.close()
    if not count:
        raise ValueError("Empty trajectory")
    return dict(frames=count, atoms=len(previous_ids), frame_interval_ps=None if step_gap is None else step_gap*dt_ps,
                output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("analyze")
    check.add_argument("input", type=Path)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--start-step", type=int, default=0)
    check.add_argument("--stride", type=int, default=1)
    conv = sub.add_parser("convert")
    conv.add_argument("input", type=Path)
    conv.add_argument("output", type=Path)
    conv.add_argument("--dt-ps", type=float, default=0.001)
    args = parser.parse_args()
    if args.command == "analyze":
        if args.stride < 1:
            parser.error("stride must be positive")
        result = analyze(args.input, args.output, start_step=args.start_step, stride=args.stride)
    else:
        if args.dt_ps <= 0:
            parser.error("dt-ps must be positive")
        result = convert(args.input, args.output, args.dt_ps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
