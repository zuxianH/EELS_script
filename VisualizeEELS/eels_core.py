"""Notebook-compatible EELS extraction, independent of the user interface."""
from dataclasses import dataclass
from pathlib import Path
import json
import io
import csv

from scan_reader import ScanReader

import numpy as np
from scipy.constants import physical_constants
from scipy.ndimage import gaussian_filter1d


KB_MEV_PER_K = physical_constants["Boltzmann constant in eV/K"][0] * 1000.0


def detailed_balance_factor(energy_mev, temperature_k):
    """Return beta*E / (1 - exp(-beta*E)), with positive E denoting loss.

    Energy is in meV and temperature in kelvin. The zero-energy limit is 1.
    Separate gain/loss branches avoid exponential overflow at low temperature.
    """
    energy = np.asarray(energy_mev, dtype=float)
    if energy.ndim != 1 or not energy.size or not np.isfinite(energy).all():
        raise ValueError("Detailed balance requires a nonempty, finite 1D energy axis")
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Detailed-balance temperature must be finite and positive (K)")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = energy / (KB_MEV_PER_K * temperature_k)
    if not np.isfinite(x).all():
        raise ValueError("Energy / (k_B T) exceeds the supported numeric range")
    magnitude = np.abs(x)
    factor = np.ones_like(x)
    nonzero = magnitude > 0
    # expm1 retains precision near zero; its argument is always nonpositive.
    with np.errstate(under="ignore"):
        factor[nonzero] = magnitude[nonzero] / -np.expm1(-magnitude[nonzero])
        gain = x < 0
        factor[gain] *= np.exp(x[gain])
    return factor


@dataclass(frozen=True)
class ScanInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str
    mtime_ns: int
    size: int

    @property
    def canonical_shape(self):
        if len(self.shape) == 3:
            energy, px, py = self.shape
            return (1, energy, 1, 1, px, py)
        return self.shape


def inspect_scan(path):
    path = Path(path).expanduser().resolve()
    scan = ScanReader(path)
    stat = path.stat()
    return ScanInfo(str(path), scan.shape, str(scan.dtype), stat.st_mtime_ns, stat.st_size)


def energy_loss_axis_mev(chunk, timestep_fs=2.5, stride=3):
    if not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError("Energy bin count must be a positive integer")
    if not np.isfinite(timestep_fs) or timestep_fs <= 0:
        raise ValueError("Time step must be finite and positive")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("Stride must be a positive integer")
    frequency_thz = np.fft.fftshift(np.fft.fftfreq(chunk, timestep_fs * stride / 1000.0))
    return frequency_thz * 4.13566769692386


def circular_detector_mask(shape, center=None, radius=20):
    if len(shape) != 2 or any(n < 1 for n in shape):
        raise ValueError("Detector plane must have two nonempty dimensions")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Detector radius must be finite and positive")
    center = center if center is not None else (shape[0] // 2, shape[1] // 2)
    if len(center) != 2 or not all(np.isfinite(v) and 0 <= v < n for v, n in zip(center, shape)):
        raise ValueError(f"Detector center {center} is outside shape {shape}")
    px, py = np.ogrid[:shape[0], :shape[1]]
    mask = (px - center[0]) ** 2 + (py - center[1]) ** 2 <= radius ** 2
    if not mask.any():
        raise ValueError("Detector contains no pixel centers; increase its radius")
    return mask


def detector_center(info, offset_px=0, offset_py=0):
    px, py = info.shape[-2:]
    return px // 2 + offset_px, py // 2 + offset_py


def detector_offsets_from_click(shape, px, py, selected_shapes):
    """Convert a preview pixel to shared offsets, checking every selected plane."""
    if not all(np.isfinite(v) and 0 <= v < n for v, n in zip((px, py), shape)):
        raise ValueError("Click inside the diffraction plane")
    offsets = tuple(int(np.floor(v + 0.5)) - n // 2 for v, n in zip((px, py), shape))
    if any(not 0 <= n // 2 + offset < n for plane in selected_shapes for n, offset in zip(plane, offsets)):
        raise ValueError("This position is outside another selected scan. Choose a point closer to the center or select scans with compatible detector planes.")
    return offsets


def gaussian_broaden_spectrum(energy_mev, intensity, sigma_mev=0.0):
    """Gaussian convolution of linear intensity on a uniform energy grid.

    Sigma is the standard deviation in meV. Reflecting boundaries preserve
    the sum of the recorded samples and do not wrap energy gain into loss.
    The kernel is truncated at four standard deviations.
    """
    energy = np.asarray(energy_mev, dtype=float)
    values = np.asarray(intensity, dtype=float)
    if energy.ndim != 1 or values.shape != energy.shape or not energy.size:
        raise ValueError("Broadening requires matching nonempty 1D energy and intensity arrays")
    if not np.isfinite(energy).all() or not np.isfinite(values).all():
        raise ValueError("Broadening requires finite energy and intensity values")
    if not np.isfinite(sigma_mev) or sigma_mev < 0:
        raise ValueError("Gaussian sigma must be finite and nonnegative")
    if sigma_mev == 0:
        return values.copy()
    if len(energy) < 2:
        raise ValueError("Gaussian broadening requires at least two energy bins")
    steps = np.diff(energy)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-8, atol=0):
        raise ValueError("Gaussian broadening requires a uniformly spaced, increasing energy axis")
    if sigma_mev > energy[-1] - energy[0]:
        raise ValueError("Gaussian sigma must not exceed the recorded energy span")
    return gaussian_filter1d(values, sigma=sigma_mev / steps[0], mode="reflect", truncate=4.0)


def _open_scan(info):
    stat = Path(info.path).stat()
    if (stat.st_mtime_ns, stat.st_size) != (info.mtime_ns, info.size):
        raise ValueError("File changed on disk; refresh the file list")
    return ScanReader(info.path)


def _validate_index(name, value, length):
    if not isinstance(value, (int, np.integer)) or not 0 <= value < length:
        raise ValueError(f"{name} index {value} is outside [0, {length})")


def _probe_key(info, sample, probe_x, probe_y):
    if len(info.shape) == 3:
        if (sample, probe_x, probe_y) != (0, 0, 0):
            raise ValueError("3D arrays contain only sample 0 and probe (0, 0)")
        return [slice(None)] * 3
    for name, value, length in zip(
        ("sample", "probe x", "probe y"), (sample, probe_x, probe_y),
        (info.shape[0], info.shape[2], info.shape[3]),
    ):
        _validate_index(name, value, length)
    return [sample, slice(None), probe_x, probe_y, slice(None), slice(None)]


def extract_spectrum(info, *, sample=0, probe_x=0, probe_y=0, radius=20,
                     offset_px=0, offset_py=0, normalize_3d=True):
    """Integrate bounded direct reads with float64 accumulation.

    Normalization divides by the sum of the entire selected (energy, px, py)
    block, exactly the normalization defined in STEM-EELS.ipynb.
    """
    mask = circular_detector_mask(info.shape[-2:], detector_center(info, offset_px, offset_py), radius)
    key = _probe_key(info, sample, probe_x, probe_y)
    energy_axis = 0 if len(info.shape) == 3 else 1
    n_energy = info.canonical_shape[1]
    spectrum = np.empty(n_energy, dtype=np.float64)
    total = 0.0
    plane_bytes = int(np.prod(info.shape[-2:])) * np.dtype(info.dtype).itemsize
    block_size = max(1, min(32, 16 * 2**20 // plane_bytes))
    with _open_scan(info) as reader:
        for start in range(0, n_energy, block_size):
            key[energy_axis] = slice(start, min(start + block_size, n_energy))
            part = reader.read(key)
            selected = part[:, mask]
            if not np.isfinite(selected).all():
                raise ValueError("Selected detector data contains NaN or infinite intensities")
            spectrum[start:start + len(part)] = selected.sum(axis=1, dtype=np.float64)
            if normalize_3d:
                if not np.isfinite(part).all():
                    raise ValueError("Cannot normalize: probe data contains NaN or infinite intensities")
                total += part.sum(dtype=np.float64)
    if normalize_3d:
        if not np.isfinite(total) or total == 0:
            raise ValueError(f"Cannot normalize probe data with total intensity {total}")
        spectrum /= total
    if not np.isfinite(spectrum).all():
        raise ValueError("Integrated intensity overflowed; check the input data")
    return spectrum


def diffraction_pattern(info, energy_index, *, sample=0, probe_x=0, probe_y=0):
    _validate_index("Energy", energy_index, info.canonical_shape[1])
    key = _probe_key(info, sample, probe_x, probe_y)
    key[0 if len(info.shape) == 3 else 1] = energy_index
    with _open_scan(info) as reader:
        return reader.read(key).astype(float)


def detector_scan_map(info, energy_index, *, sample=0, radius=21,
                      offset_px=0, offset_py=0):
    """Return raw detector sums indexed by (probe_x, probe_y) at one energy.

    Read at most one probe-x row at a time, splitting larger rows into blocks
    of at most 16 MiB (or one detector plane if a plane exceeds that size).
    No spectrum normalization, detailed balance, or broadening is applied.
    """
    if len(info.shape) != 6:
        raise ValueError("A 2D scan map requires a 6D scan")
    _validate_index("Sample", sample, info.shape[0])
    _validate_index("Energy", energy_index, info.shape[1])
    mask = circular_detector_mask(info.shape[-2:], detector_center(info, offset_px, offset_py), radius)
    n_x, n_y = info.shape[2:4]
    scan_map = np.empty((n_x, n_y), dtype=np.float64)
    plane_bytes = int(np.prod(info.shape[-2:])) * np.dtype(info.dtype).itemsize
    row_step = max(1, 16 * 2**20 // plane_bytes)
    with _open_scan(info) as reader:
        for x in range(n_x):
            for y in range(0, n_y, row_step):
                stop = min(y + row_step, n_y)
                row = reader.read((sample, energy_index, x, slice(y, stop), slice(None), slice(None)))
                values = np.sum(row, axis=(-2, -1), where=mask, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("Selected detector data contains NaN, infinite, or overflowed intensities")
                scan_map[x, y:stop] = values
    return scan_map


def display_intensity(values, mode):
    if mode == "log10":
        return np.log10(np.clip(values, np.finfo(float).tiny, None))
    return values


def export_csv(curves):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["label", "source_file", "sample", "probe_x", "probe_y", "energy_meV", "intensity"])
    for curve in curves:
        for energy, intensity in zip(curve["energy"], curve["intensity"]):
            writer.writerow([curve["label"], curve["path"], curve["sample"], curve["probe_x"],
                             curve["probe_y"], energy, intensity])
    return output.getvalue().encode("utf-8")


def export_npz(curves, settings):
    arrays = {f"curve_{i:03d}": np.column_stack((c["energy"], c["intensity"])) for i, c in enumerate(curves)}
    metadata = {"settings": settings, "curves": [
        {k: v for k, v in c.items() if k not in ("energy", "intensity")} for c in curves
    ]}
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()
