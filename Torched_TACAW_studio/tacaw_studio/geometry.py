"""Scan geometry in Angstrom, with TACAW's native i/N convention."""
from __future__ import annotations
import numpy as np


def scan_parameters(scan: dict) -> dict:
    kind = scan["kind"]
    if kind == "Plane wave":
        return {"beam_conv_ang_mrad": 0.0, "scanning_mode": "unitcell",
                "scanning_shape": [1, 1], "scanning_batch_shape": [1, 1]}
    shape = [int(n) for n in scan["shape"]]
    batch = [int(n) for n in scan["batch"]]
    if kind == "Line scan":
        shape[1], batch[1] = 1, 1
    if len(shape) != 2 or len(batch) != 2 or min(shape + batch) < 1:
        raise ValueError("Scan and batch shapes must contain two positive integers.")
    if any(n % b for n, b in zip(shape, batch)):
        raise ValueError("Each batch dimension must divide its scan dimension exactly.")
    if kind == "Unit cell":
        return {"scanning_mode": "unitcell", "scanning_shape": shape,
                "scanning_batch_shape": batch}
    if kind == "Basis vectors":
        origin = np.asarray(scan["start"], dtype=float)
        basis = np.asarray(scan["basis"], dtype=float)
    else:
        origin = np.asarray(scan["start"], dtype=float)
        end = np.asarray(scan["end"], dtype=float)
        if kind == "Line scan":
            if shape[0] < 2 or np.allclose(origin, end):
                raise ValueError("A line needs distinct endpoints and at least two points.")
            basis = np.array([end - origin, [0., 1.]])
        elif kind == "2D scan":
            problems = []
            for axis, label in enumerate(("x", "y")):
                if np.isclose(end[axis] - origin[axis], 0):
                    problems.append(f"End {label} equals Start {label}; choose a different End {label} coordinate")
                if shape[axis] < 2:
                    problems.append(f"Scan points {label} is {shape[axis]}; set it to at least 2")
            if problems:
                raise ValueError("For a 2D scan, select opposite rectangle corners. " + "; ".join(problems) + ".")
            basis = np.diag(end - origin)
        else:
            raise ValueError(f"Unknown scan kind: {kind}")
        if not scan.get("periodic", False):
            for axis, n in enumerate(shape):
                if n > 1:
                    basis[axis] *= n / (n - 1)
    if origin.shape != (2,) or basis.shape != (2, 2):
        raise ValueError("Origin must be [x,y]; basis must contain two [x,y] vectors.")
    if not np.isfinite(origin).all() or not np.isfinite(basis).all():
        raise ValueError("Scan coordinates must be finite.")
    if shape[0] > 1 and np.linalg.norm(basis[0]) == 0:
        raise ValueError("First scan vector cannot be zero.")
    if shape[1] > 1 and np.linalg.norm(basis[1]) == 0:
        raise ValueError("Second scan vector cannot be zero.")
    if min(shape) > 1 and abs(np.linalg.det(basis)) < 1e-12:
        raise ValueError("A 2D scan needs two independent basis vectors.")
    return {"scanning_mode": "parallelogram", "scanning_shape": shape,
            "scanning_batch_shape": batch, "scanning_origin": origin.tolist(),
            "scanning_basis_vectors": basis.tolist()}


def coordinates(parameters: dict, cell: np.ndarray | None = None) -> np.ndarray:
    """Return (nx, ny, 2) coordinates in engine ij axis order."""
    shape = parameters["scanning_shape"]
    mode = parameters.get("scanning_mode", "unitcell")
    origin = np.asarray(parameters.get("scanning_origin", [0., 0.]), float)
    if mode == "unitcell":
        if cell is None:
            raise ValueError("Unit-cell preview needs a trajectory frame.")
        basis, origin = np.asarray(cell)[:2, :2], np.zeros(2)
    elif mode == "parallelogram":
        basis = np.asarray(parameters["scanning_basis_vectors"], float)
    elif mode in ("box", "box_relative"):
        size = parameters.get("scanning_box_size")
        if size is None:
            size = np.asarray(parameters["scanning_box_end"]) - origin
        basis = np.diag(size)
        if mode == "box_relative":
            if cell is None:
                raise ValueError("Relative-box preview needs a trajectory frame.")
            dims = np.diag(cell)[:2]
            origin, basis = origin * dims, basis * dims[:, None]
    else:
        raise ValueError(f"Unsupported scanning mode: {mode}")
    grid = np.stack(np.meshgrid(np.arange(shape[0]) / shape[0],
                               np.arange(shape[1]) / shape[1], indexing="ij"), axis=-1)
    return grid @ basis + origin


def scan_from_parameters(p: dict) -> dict:
    mode = p.get("scanning_mode", "unitcell")
    kind = "Basis vectors" if mode == "parallelogram" else "Unit cell"
    if mode in ("box", "box_relative"):
        kind = "Native box"
    if float(p.get("beam_conv_ang_mrad", 20)) == 0 and p.get("scanning_shape") == [1, 1]:
        kind = "Plane wave"
    return {"kind": kind, "shape": list(p.get("scanning_shape", [5, 1])),
            "batch": list(p.get("scanning_batch_shape", [1, 1])),
            "start": list(p.get("scanning_origin", [0., 0.])), "end": [1., 1.],
            "basis": p.get("scanning_basis_vectors", [[1., 0.], [0., 1.]]),
            "periodic": False}
