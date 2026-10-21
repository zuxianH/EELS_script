"""Bounded trajectory discovery, reading, and atomic conversion."""
from __future__ import annotations

from contextlib import ExitStack
from io import StringIO
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import iread, read, write
from ase.io.trajectory import Trajectory
from ase.units import Bohr


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def discover(folder: str | Path, recursive=False) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"Input folder does not exist: {root}")
    found = []
    for directory, subdirs, names in os.walk(root, followlinks=False):
        subdirs[:] = [n for n in subdirs if not n.startswith(".") and
                      n != "tacaw_runs" and not n.endswith((".zarray", ".zarr")) and
                      not (Path(directory)/n/"studio.yaml").exists()]
        for name in names:
            p = Path(directory)/name
            if p.suffix.lower() in (".traj", ".xyz", ".extxyz", ".lammps", ".dump", ".lammpstrj"):
                found.append(p.absolute())
        if not recursive:
            break
    return sorted(found, key=lambda p: (p.suffix.lower() != ".traj", natural_key(p)))


def scale(unit: str | None) -> float:
    if unit is None or unit.lower() in ("auto", "angstrom", "angstroms", "ang", "a", "ase"):
        return 1.
    if unit.lower().replace("-", "_") in ("bohr", "atomic_unit", "atomic_units", "au", "a0"):
        return Bohr
    raise ValueError(f"Unsupported length unit: {unit}")


def xyz_frames(path: Path, options: dict):
    """Plain/i-PI XYZ reader; one frame at a time, with independent cell units."""
    with path.open() as handle:
        frame = 0
        while True:
            count = handle.readline()
            while count and not count.strip():
                count = handle.readline()
            if not count:
                return
            try:
                n = int(count)
            except ValueError as exc:
                raise ValueError(f"Frame {frame}: invalid atom count.") from exc
            if n < 1:
                raise ValueError(f"Frame {frame}: atom count must be positive.")
            comment = handle.readline()
            if not comment:
                raise ValueError(f"Frame {frame}: missing comment.")
            symbols, positions = [], np.empty((n, 3))
            for i in range(n):
                fields = handle.readline().split()
                if len(fields) < 4:
                    raise ValueError(f"Frame {frame}, atom {i}: incomplete XYZ.")
                symbols.append(fields[0])
                positions[i] = [float(x) for x in fields[1:4]]
            position_match = re.search(r"(?:positions|x_centroid|x)\{([^}]+)\}", comment, re.I)
            cell_match = re.search(r"cell\{([^}]+)\}", comment, re.I)
            pu = options["positions_unit"]
            cu = options["cell_unit"]
            if pu == "auto":
                pu = position_match.group(1) if position_match else None
            if cu == "auto":
                cu = cell_match.group(1) if cell_match else None
            positions *= scale(pu)
            cell, pbc = None, False
            abc = re.search(r"CELL\(abcABC\):\s*(.+?)(?:Step:|Bead:|$)", comment, re.I)
            if abc:
                fields = [float(x) for x in abc.group(1).split()[:6]]
                if len(fields) != 6:
                    raise ValueError("CELL(abcABC) needs six numbers.")
                fields[:3] = [x*scale(cu) for x in fields[:3]]
                cell, pbc = cellpar_to_cell(fields), True
            info = {"xyz_comment": comment.rstrip()}
            for key in ("step", "bead"):
                match = re.search(rf"\b{key}:\s*(-?\d+)", comment, re.I)
                if match:
                    info[key] = int(match.group(1))
            yield apply_overrides(Atoms(symbols, positions=positions, cell=cell, pbc=pbc, info=info), options)
            frame += 1


def apply_overrides(atoms: Atoms, options: dict) -> Atoms:
    cell = options.get("cell")
    if cell is not None:
        cell = np.asarray(cell, float)
        atoms.set_cell(cellpar_to_cell(cell) if cell.shape == (6,) else cell)
    if options.get("pbc") is not None:
        atoms.pbc = options["pbc"]
    if not np.isfinite(atoms.positions).all():
        raise ValueError("Trajectory contains nonfinite atom coordinates.")
    return atoms


def lammps_frames(path: Path, options: dict):
    mapping = options.get("type_symbols", {})
    specorder = None
    if mapping:
        mapping = {int(k): v for k, v in mapping.items()}
        if set(mapping) != set(range(1, max(mapping)+1)):
            raise ValueError("LAMMPS element mapping must cover consecutive types starting at 1.")
        specorder = [mapping[i] for i in range(1, max(mapping)+1)]
    # ASE's whole-file dump reader accumulates lines; feed it one timestep.
    with path.open() as handle:
        block = []
        for line in handle:
            if line.startswith("ITEM: TIMESTEP") and block:
                atoms = read(StringIO("".join(block)), format="lammps-dump-text", specorder=specorder)
                atoms.info["step"] = int(block[1].strip())
                yield standard_overrides(atoms, options)
                block = []
            block.append(line)
        if block:
            atoms = read(StringIO("".join(block)), format="lammps-dump-text", specorder=specorder)
            atoms.info["step"] = int(block[1].strip())
            yield standard_overrides(atoms, options)


def standard_overrides(atoms: Atoms, options: dict) -> Atoms:
    # ASE formats express lengths in Angstrom. Explicit overrides mean the
    # source numbers use the chosen unit, and are applied once here.
    if options["positions_unit"] != "auto":
        atoms.positions *= scale(options["positions_unit"])
    if options["cell_unit"] != "auto" and options.get("cell") is None:
        atoms.set_cell(atoms.cell.array * scale(options["cell_unit"]))
    return apply_overrides(atoms, options)


def frames(path: str | Path, options: dict):
    path = Path(path).expanduser()
    fmt = options.get("format", "auto")
    if path.suffix.lower() == ".traj":
        with Trajectory(path, mode="r") as trajectory:
            for atoms in trajectory:
                yield standard_overrides(atoms, options)
        return
    if fmt == "auto":
        if path.suffix.lower() in (".lammps", ".dump", ".lammpstrj"):
            fmt = "lammps-dump-text"
        else:
            with path.open() as h:
                h.readline()
                comment = h.readline()
            fmt = "extxyz" if re.search(r'\b(?:Lattice|Properties)\s*=', comment) else "xyz"
    if fmt in ("xyz", "ipi"):
        yield from xyz_frames(path, options)
    elif fmt == "lammps-dump-text":
        yield from lammps_frames(path, options)
    elif fmt == "extxyz":
        for atoms in iread(path, index=":", format="extxyz"):
            yield standard_overrides(atoms, options)
    else:
        raise ValueError(f"Unsupported trajectory format: {fmt}")


def preview(path: str, index: int, options: dict) -> tuple[Atoms, int | None]:
    p = Path(path).expanduser()
    if p.suffix.lower() == ".traj":
        with Trajectory(p) as t:
            if not -len(t) <= index < len(t):
                raise ValueError(f"Frame index outside trajectory of {len(t)} frames.")
            return standard_overrides(t[index], options), len(t)
    if index < 0:
        raise ValueError("Negative preview indices require an indexed .traj file.")
    iterator = frames(p, options)
    try:
        for i, atoms in enumerate(iterator):
            if i == index:
                return atoms, None
    finally:
        iterator.close()
    raise ValueError(f"Source has no frame {index}.")


def supported_cell(atoms: Atoms) -> None:
    c = atoms.cell.array
    if len(atoms) < 1 or not np.isfinite(c).all() or np.any(np.diag(c) <= 0):
        raise ValueError("TACAW needs atoms and a nonzero positive simulation cell. Set a cell override if absent.")
    if not np.allclose(c, np.diag(np.diag(c)), atol=1e-7):
        raise ValueError("The current TACAW engine uses diagonal cells. Rotate/rebuild this structure into an axis-aligned orthorhombic cell.")
    if not atoms.pbc.all():
        raise ValueError("TACAW's multislice requires periodic boundaries in all three axes. Set PBC explicitly if appropriate.")


def needs_conversion(path: str, options: dict) -> bool:
    return (Path(path).suffix.lower() != ".traj" or options["start"] != 0 or
            options["stop"] is not None or options["stride"] != 1 or
            options.get("cell") is not None or options.get("pbc") is not None or
            options["positions_unit"] != "auto" or options["cell_unit"] != "auto")


def convert(source: str, target: Path, options: dict, report=print) -> int:
    if target.exists():
        raise FileExistsError(f"Converted trajectory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.stem+".", suffix=".partial.traj", dir=target.parent)
    os.close(fd)
    partial = Path(temporary)
    iterator = frames(source, options)
    written = 0
    reference_numbers = None
    try:
        with Trajectory(partial, mode="w") as output:
            for i, atoms in enumerate(iterator):
                if options["stop"] is not None and i >= options["stop"]:
                    break
                if i < options["start"] or (i-options["start"]) % options["stride"]:
                    continue
                supported_cell(atoms)
                if reference_numbers is None:
                    reference_numbers = atoms.numbers.copy()
                elif not np.array_equal(reference_numbers, atoms.numbers):
                    raise ValueError(f"Atom ordering/count changes at source frame {i}.")
                output.write(atoms)
                written += 1
                if written == 1 or written % 100 == 0:
                    report(f"Converted {written} frames; source index {i}", flush=True)
        if not written:
            raise ValueError("Frame selection produced an empty trajectory.")
        # Never replace an existing destination, even in another preparation.
        os.link(partial, target)
        return written
    finally:
        iterator.close()
        partial.unlink(missing_ok=True)


def compare_frames(atoms: list[Atoms], index: int):
    reference = atoms[0]
    supported_cell(reference)
    for bead, a in enumerate(atoms[1:], 1):
        supported_cell(a)
        if not np.array_equal(a.numbers, reference.numbers):
            raise ValueError(f"Bead {bead}: atom ordering/count mismatch at frame {index}.")
        if not np.allclose(a.cell.array, reference.cell.array, rtol=1e-7, atol=1e-7):
            raise ValueError(f"Bead {bead}: simulation cell mismatch at frame {index}.")
        for key in ("step", "time", "timestep"):
            values = [b.info.get(key) for b in atoms]
            if any(v is not None for v in values) and any(v != values[0] for v in values):
                raise ValueError(f"Beads have mismatched {key} metadata at frame {index}.")


def validate_beads(paths: list[str], indices=None) -> dict:
    with ExitStack() as stack:
        readers = [stack.enter_context(Trajectory(p)) for p in paths]
        lengths = [len(r) for r in readers]
        if not lengths or min(lengths) < 1:
            raise ValueError("Trajectory has no frames.")
        if len(set(lengths)) != 1:
            raise ValueError(f"Bead frame counts differ: {lengths}")
        sample = sorted(set([0, lengths[0]-1]) if indices is None else set(indices))
        for index in sample:
            compare_frames([r[index] for r in readers], index)
        first = readers[0][0]
        return {"frames": lengths[0], "atoms": len(first), "beads": len(paths),
                "cell": first.cell.array.tolist(),
                "time_metadata_available": any(k in first.info for k in ("step", "time", "timestep"))}


def write_structure(path: str, output: Path):
    with Trajectory(path) as reader:
        write(output, reader[0], format="extxyz")


def displayed_atoms(atoms: Atoms, z_min: float, z_max: float, repeats: int | None = None):
    positions = atoms.positions.copy()
    # Match the notebook's wrapped projection without altering calculation data.
    if atoms.cell.rank == 3 and atoms.pbc.all():
        copy = atoms.copy()
        copy.wrap()
        positions = copy.positions
    if repeats:
        z_max = min(z_max, atoms.cell.lengths()[2] / repeats)
    indices = np.flatnonzero((positions[:, 2] >= z_min) & (positions[:, 2] < z_max))
    return positions[indices], atoms.numbers[indices], indices
