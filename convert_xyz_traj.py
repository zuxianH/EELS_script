#!/usr/bin/env python3
"""Stream an XYZ or i-PI XYZ trajectory into an ASE .traj file.

With only an input argument, the output is placed beside the input and its
final suffix is replaced by ".traj":

    python convert_xyz_traj.py /path/to/simulation.pos_0.xyz

Unlike ase.io.read(path, index=":"), this converter keeps only one frame in
memory. It also preserves i-PI cell/unit annotations.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io.trajectory import Trajectory
from ase.units import Bohr


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
CELL_FIELDS = r"\s+".join([f"({FLOAT})"] * 6)
CELL_RE = re.compile(
    rf"CELL\(abcABC\):\s*{CELL_FIELDS}",
    re.IGNORECASE,
)
POSITION_UNIT_RE = re.compile(r"positions\{([^}]+)\}", re.IGNORECASE)
CELL_UNIT_RE = re.compile(r"cell\{([^}]+)\}", re.IGNORECASE)
STEP_RE = re.compile(r"\bStep:\s*(-?\d+)", re.IGNORECASE)
BEAD_RE = re.compile(r"\bBead:\s*(-?\d+)", re.IGNORECASE)


def unit_scale(unit: str | None, *, quantity: str) -> float:
    """Return the scale that converts an i-PI unit annotation to Angstrom."""
    if unit is None:
        return 1.0

    normalized = unit.strip().lower().replace("-", "_")
    if normalized in {"angstrom", "angstroms", "ang", "a"}:
        return 1.0
    if normalized in {"atomic_unit", "atomic_units", "bohr", "a0", "au"}:
        return Bohr
    raise ValueError(f"unsupported {quantity} unit {unit!r}")


def parse_comment(comment: str) -> tuple[np.ndarray | None, bool, float, dict]:
    """Extract cell, PBC, position scale, and useful metadata from a comment."""
    info: dict[str, int | str] = {"xyz_comment": comment.rstrip("\n")}

    step_match = STEP_RE.search(comment)
    bead_match = BEAD_RE.search(comment)
    if step_match:
        info["step"] = int(step_match.group(1))
    if bead_match:
        info["bead"] = int(bead_match.group(1))

    position_match = POSITION_UNIT_RE.search(comment)
    position_scale = unit_scale(
        position_match.group(1) if position_match else None,
        quantity="position",
    )

    cell_match = CELL_RE.search(comment)
    if not cell_match:
        return None, False, position_scale, info

    cellpar = np.asarray(cell_match.groups(), dtype=float)
    cell_unit_match = CELL_UNIT_RE.search(comment)
    cellpar[:3] *= unit_scale(
        cell_unit_match.group(1) if cell_unit_match else None,
        quantity="cell",
    )
    return cellpar_to_cell(cellpar), True, position_scale, info


def read_xyz_frames(path: Path) -> Iterator[Atoms]:
    """Yield frames from a regular or i-PI-style XYZ file one at a time."""
    with path.open("r", encoding="utf-8") as handle:
        frame_index = 0
        while True:
            count_line = handle.readline()
            while count_line and not count_line.strip():
                count_line = handle.readline()
            if not count_line:
                return

            try:
                atom_count = int(count_line)
            except ValueError as exc:
                raise ValueError(
                    f"frame {frame_index}: expected atom count, "
                    f"got {count_line.rstrip()!r}"
                ) from exc
            if atom_count < 0:
                raise ValueError(
                    f"frame {frame_index}: negative atom count {atom_count}"
                )

            comment = handle.readline()
            if not comment:
                raise ValueError(f"frame {frame_index}: missing XYZ comment line")

            symbols: list[str] = []
            positions = np.empty((atom_count, 3), dtype=float)
            for atom_index in range(atom_count):
                line = handle.readline()
                if not line:
                    raise ValueError(
                        f"frame {frame_index}: file ended after {atom_index} of "
                        f"{atom_count} atoms"
                    )
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(
                        f"frame {frame_index}, atom {atom_index}: expected symbol "
                        f"and three coordinates, got {line.rstrip()!r}"
                    )
                symbols.append(fields[0])
                try:
                    positions[atom_index] = fields[1:4]
                except ValueError as exc:
                    raise ValueError(
                        f"frame {frame_index}, atom {atom_index}: invalid coordinates"
                    ) from exc

            cell, pbc, position_scale, info = parse_comment(comment)
            positions *= position_scale
            yield Atoms(
                symbols,
                positions=positions,
                cell=cell,
                pbc=pbc,
                info=info,
            )
            frame_index += 1


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input XYZ trajectory")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="output ASE trajectory (default: input suffix replaced by .traj)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="first frame to write (default: 0)",
    )
    parser.add_argument(
        "--stop",
        type=int,
        help="stop before this frame (default: end of file)",
    )
    parser.add_argument(
        "--stride",
        type=positive_integer,
        default=1,
        help="write every Nth frame after --start (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    parser.add_argument(
        "--progress-every",
        type=positive_integer,
        default=100,
        metavar="N",
        help="print progress every N written frames (default: 100)",
    )
    args = parser.parse_args(argv)

    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.stop is not None and args.stop < args.start:
        parser.error("--stop must be greater than or equal to --start")
    return args


def convert(args: argparse.Namespace) -> tuple[Path, int]:
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_suffix(".traj")
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("input and output paths must be different")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output_path} "
            "(use --overwrite to replace it)"
        )
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"output directory does not exist: {output_path.parent}"
        )

    written = 0
    try:
        with Trajectory(output_path, mode="w") as trajectory:
            for frame_index, atoms in enumerate(read_xyz_frames(input_path)):
                if args.stop is not None and frame_index >= args.stop:
                    break
                if (
                    frame_index < args.start
                    or (frame_index - args.start) % args.stride
                ):
                    continue
                trajectory.write(atoms)
                written += 1
                if written == 1 or written % args.progress_every == 0:
                    print(
                        f"Wrote {written} frame(s); source frame {frame_index}",
                        file=sys.stderr,
                        flush=True,
                    )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    print(
        f"Finished: wrote {written} frame(s) to {output_path}",
        file=sys.stderr,
    )
    return output_path, written


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        convert(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
