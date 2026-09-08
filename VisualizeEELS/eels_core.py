"""Notebook-compatible EELS extraction, independent of the user interface."""
from dataclasses import dataclass
from pathlib import Path
import json
import io
import csv

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
    scan = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if not isinstance(scan, np.ndarray):
            raise ValueError("Expected a .npy array, not an archive")
        if scan.ndim not in (3, 6) or any(n == 0 for n in scan.shape):
            raise ValueError(
                f"Expected (energy, px, py) or (sample, energy, probe_x, probe_y, px, py); got {scan.shape}"
            )
        if scan.dtype.kind not in "fiu":
            raise ValueError(f"Expected real numeric intensities; got {scan.dtype}")
        stat = path.stat()
        return ScanInfo(str(path), scan.shape, str(scan.dtype), stat.st_mtime_ns, stat.st_size)
    finally:
        if hasattr(scan, "close"):
            scan.close()
        del scan


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


def _open_probe(info, sample, probe_x, probe_y):
    stat = Path(info.path).stat()
    if (stat.st_mtime_ns, stat.st_size) != (info.mtime_ns, info.size):
        raise ValueError("File changed on disk; refresh the file list")
    scan = np.load(info.path, mmap_mode="r", allow_pickle=False)
    if len(info.shape) == 3:
        if (sample, probe_x, probe_y) != (0, 0, 0):
            raise ValueError("3D arrays contain only sample 0 and probe (0, 0)")
        return scan
    for name, value, length in zip(
        ("sample", "probe x", "probe y"), (sample, probe_x, probe_y),
        (info.shape[0], info.shape[2], info.shape[3]),
    ):
        if not isinstance(value, (int, np.integer)) or not 0 <= value < length:
            raise ValueError(f"{name} index {value} is outside [0, {length})")
    return scan[sample, :, probe_x, probe_y, :, :]


def extract_spectrum(info, *, sample=0, probe_x=0, probe_y=0, radius=20,
                     offset_px=0, offset_py=0, normalize_3d=True):
    """Integrate with bounded working memory and float64 accumulation.

    Normalization divides by the sum of the entire selected (energy, px, py)
    block, exactly the normalization defined in STEM-EELS.ipynb.
    """
    mask = circular_detector_mask(info.shape[-2:], detector_center(info, offset_px, offset_py), radius)
    block = _open_probe(info, sample, probe_x, probe_y)
    spectrum = np.empty(block.shape[0], dtype=np.float64)
    total = 0.0
    for start in range(0, block.shape[0], 32):
        part = block[start:start + 32]
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
    if not 0 <= energy_index < info.canonical_shape[1]:
        raise ValueError("Energy index is outside this scan")
    return np.array(_open_probe(info, sample, probe_x, probe_y)[energy_index], dtype=float)


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
