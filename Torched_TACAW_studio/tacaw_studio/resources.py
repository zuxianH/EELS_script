"""Disk and working-array estimates for the supplied engine; no Torch imports."""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil

import numpy as np

from .config import effective_parameters, validate_parameters

GIB = 1024 ** 3


def roi_shape(parameters: dict, cell_lengths=None) -> list[int]:
    full = np.asarray(parameters["kspace_shape_full"], int)
    limited = (full * parameters["kspace_bandwidth_limiting"]).astype(int)
    mode = parameters["kspace_ROI_mode"]
    if mode in (None, "center"):
        shape = limited if mode is None or parameters.get("kspace_ROI_shape") is None else np.asarray(parameters["kspace_ROI_shape"], int)
    else:
        if cell_lengths is None:
            raise ValueError("Select a trajectory with a cell to estimate the reciprocal-space ROI.")
        lengths = np.asarray(cell_lengths, float) * parameters["sample_supercell_int"]
        if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("Resource estimates require positive cell lengths.")
        # Same relativistic electron wavenumber and inclusive FFT bins as Config.
        h, mass, charge, light = 6.62607015e-34, 9.1093837139e-31, 1.602176634e-19, 299792458.
        voltage = parameters["beam_energy_keV"] * 1000
        wavenumber = math.sqrt(2 * mass * charge * voltage * (1 + charge * voltage / (2 * mass * light ** 2))) / h * 1e-10
        shape = []
        for axis, n in enumerate(full):
            angles = np.fft.fftshift(np.fft.fftfreq(n, lengths[axis] / n)) / wavenumber * 1000
            indices = np.flatnonzero((angles >= parameters["kspace_ROI_min"][axis]) &
                                     (angles <= parameters["kspace_ROI_max"][axis]))
            if not indices.size:
                raise ValueError("Reciprocal-space ROI contains no grid bins; adjust its bounds.")
            bandwidth_start = math.ceil((n - limited[axis]) / 2)
            if indices[0] < bandwidth_start or indices[-1] >= bandwidth_start + limited[axis]:
                raise ValueError("Reciprocal-space ROI exceeds the propagated bandwidth-limited grid.")
            shape.append(int(indices.size))
    if min(shape) < 1 or np.any(np.asarray(shape) > limited):
        raise ValueError("Reciprocal-space ROI is empty or exceeds the propagated grid.")
    return [int(n) for n in shape]


def frequency_bins(parameters: dict, conversion_stride=1) -> int:
    n = parameters["trajectory_chunks_size"]
    dt = parameters["trajectory_timestep_fs"] * conversion_stride * parameters["trajectory_chunks_step"] / 1000
    step = 1. / (n * dt)
    lo, hi = parameters["frequency_THz_ROI"]
    # Count integer FFT indices without allocating a full time-axis array.
    first = max(-(n // 2), math.ceil(lo / step))
    last = min((n - 1) // 2, math.floor(hi / step))
    # Correct boundary roundoff with NumPy's actual fftfreq multiplication order.
    for _ in range(2):
        if first > -(n // 2) and (first - 1) * step >= lo:
            first -= 1
        if first * step < lo:
            first += 1
        if last < (n - 1) // 2 and (last + 1) * step <= hi:
            last += 1
        if last * step > hi:
            last -= 1
    count = max(0, last - first + 1)
    if count < 1:
        raise ValueError("Frequency ROI contains no FFT bins; adjust its bounds or chunk size.")
    return count


def estimate_resources(settings: dict, *, cell_lengths=None, atom_count=None,
                       element_count=None, source_frames=None, engine_config=None,
                       array_info=None) -> dict:
    p = effective_parameters(settings)
    validate_parameters(p)
    if engine_config is not None:
        # Saved derived axes are authoritative for a prepared run.
        sim, beam, traj = engine_config["simulation"], engine_config["beam"], engine_config["trajectory"]
        full = sim["kspace"]["shape_full"]
        roi = sim["kspace"]["ROI_shape"]
        energies = sim["frequency_THz"]["ROI_len"]
        scan, batch = beam["scanning"]["shape"], beam["scanning"]["batch_shape"]
        samples, slices = traj["chunks"]["size"], sim["n_slices"]
    else:
        full, roi = p["kspace_shape_full"], roi_shape(p, cell_lengths)
        energies = frequency_bins(p, settings["inputs"]["conversion"]["stride"])
        scan, batch = p["scanning_shape"], p["scanning_batch_shape"]
        samples, slices = p["trajectory_chunks_size"], p["n_slices"]
    pixels, roi_pixels, probes = math.prod(full), math.prod(roi), math.prod(batch)
    shape, chunks = [1, energies, *scan, *roi], [1, energies, *batch, *roi]
    output_bytes = math.prod(shape) * 8
    chunk_bytes = math.prod(chunks) * 8
    if array_info is not None:
        shape, chunks = array_info["shape"], array_info["chunks"]
        output_bytes, chunk_bytes = array_info["bytes"], array_info["chunk_bytes"]
    # TACAW's retained exit waves are complex128; pyms defaults to float32 /
    # complex64 for its per-snapshot potential and multislice work arrays.
    wave_bytes = samples * probes * roi_pixels * 16
    initial_wave_bytes = probes * pixels * 16
    fft_bytes = 3 * wave_bytes
    elements = element_count or 1
    atoms_bytes = (atom_count or 0) * 256
    transmission_bytes = slices * pixels * 8
    potential_temporaries = (2 * elements + 6) * slices * pixels * 4
    multislice_bytes = wave_bytes + transmission_bytes + potential_temporaries + 3 * probes * pixels * 8 + atoms_bytes
    device_arrays = max(fft_bytes, multislice_bytes)
    host_arrays = max(3 * initial_wave_bytes + (elements + 6) * pixels * 8 + atoms_bytes,
                      initial_wave_bytes + 4 * chunk_bytes + atoms_bytes)
    backend = settings["execution"]["profile"]["backend"]
    if backend == "cpu":
        ram_arrays, gpu_arrays = host_arrays + device_arrays, 0
    else:
        ram_arrays, gpu_arrays = host_arrays, device_arrays
    # This is planning headroom, not a measured peak or an upper bound.
    ram_plan = math.ceil(1.5 * ram_arrays) + GIB
    gpu_plan = math.ceil(1.5 * gpu_arrays) + GIB if gpu_arrays else 0
    slurm = settings["execution"]["profile"]["scheduler"] == "slurm"
    workers_per_node = settings["execution"]["slurm"]["tasks_per_node"] if slurm else 1
    nodes = settings["execution"]["slurm"]["nodes"] if slurm else 1
    conversion = settings["inputs"]["conversion"]
    from .trajectories import needs_conversion
    converted_inputs = sum(needs_conversion(path, conversion) for path in settings["inputs"]["files"])
    converted_bytes = 0 if not converted_inputs else None
    if converted_inputs and source_frames is not None and atom_count is not None:
        stop = min(source_frames, conversion["stop"]) if conversion["stop"] is not None else source_frames
        count = max(0, math.ceil((stop - conversion["start"]) / conversion["stride"]))
        converted_bytes = count * atom_count * 3 * 8 * converted_inputs
    return {"shape": shape, "chunks": chunks, "dtype": array_info["dtype"] if array_info else "float64",
            "frequency_bins": energies, "roi_shape": list(roi),
            "result_bytes": output_bytes, "npy_bytes": output_bytes + 256,
            "chunk_bytes": chunk_bytes, "wave_bytes": wave_bytes,
            "initial_wave_bytes": initial_wave_bytes, "fft_bytes": fft_bytes,
            "transmission_bytes": transmission_bytes, "multislice_bytes": multislice_bytes,
            "ram_arrays_bytes": ram_arrays, "gpu_arrays_bytes": gpu_arrays,
            "ram_plan_bytes": ram_plan, "gpu_plan_bytes": gpu_plan,
            "ram_node_plan_bytes": ram_plan * workers_per_node,
            "workers_per_node": workers_per_node, "worker_count": workers_per_node * nodes,
            "converted_bytes": converted_bytes, "converted_inputs": converted_inputs,
            "element_count_known": element_count is not None,
            "prepared": engine_config is not None, "backend": backend}


def destination_space(path: Path) -> dict:
    existing = path.expanduser().absolute()
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    disk = shutil.disk_usage(existing)
    return {"path": str(existing), "free_bytes": disk.free, "total_bytes": disk.total}


def measured_run_files(folder: Path) -> dict:
    """Explicit on-demand measurement; never follow external trajectory symlinks."""
    groups = {"Zarr results": 0, "NumPy results": 0, "Converted trajectories": 0,
              "Logs / configuration / runner": 0}
    for directory, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            relative = path.relative_to(folder)
            if any(part.endswith((".zarr", ".zarray")) for part in relative.parts):
                group = "Zarr results"
            elif len(relative.parts) == 1 and path.suffix == ".npy" and not name.startswith("scan_coordinates"):
                group = "NumPy results"
            elif relative.parts[0] == "trajectories":
                group = "Converted trajectories"
            else:
                group = "Logs / configuration / runner"
            groups[group] += path.stat().st_size
    return {"groups": groups, "total_bytes": sum(groups.values())}


def node_memory() -> dict:
    """Physical host information; this is not the scheduler's job memory limit."""
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            fields[key] = int(value.split()[0]) * 1024
        return {"total_bytes": fields["MemTotal"], "available_bytes": fields.get("MemAvailable")}
    except (OSError, ValueError, KeyError):
        try:
            return {"total_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
                    "available_bytes": None}
        except (OSError, ValueError, AttributeError):
            return {}
