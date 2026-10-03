"""Streamlit caches shared across views.

Scientific calculations and readers stay independent of Streamlit. Sharing
one cached wrapper (rather than each view defining its own) also means a
diffraction plane requested from two different tabs is only read once.
"""
import numpy as np
import streamlit as st

from eels_studio.core.spectra import (
    diffraction_pattern, extract_spectrum_tile, extract_spectrum, spectrum_from_tile,
)
from eels_studio.io.scans import inspect_scan
from eels_studio.io.scan_sources import scan_revision

PNG_DPI_OPTIONS = [150, 300, 600, 1200]


@st.cache_data(show_spinner=False, max_entries=32)
def cached_diffraction_pattern(info, energy_index, dummy, probe_x, probe_y):
    return diffraction_pattern(info, energy_index, dummy=dummy, probe_x=probe_x, probe_y=probe_y)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_background_fit(energy, intensity, config):
    # Cache only small extracted 1D arrays; never open a simulation file here.
    from eels_studio.core.background import fit_background
    return fit_background(energy, intensity, config)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_spectrum_tile(info, dummy, tile_x, tile_y, radius, offset_px, offset_py):
    # Cache only reduced spectra, never the large decoded source chunks.
    return extract_spectrum_tile(info, dummy=dummy, tile_x=tile_x, tile_y=tile_y,
                                 radius=radius, offset_px=offset_px, offset_py=offset_py)


def spectrum_from_zarr_tile(info, dummy, probe_x, probe_y, radius, offset_px, offset_py, normalize):
    """Adjacent probes reuse the same small detector-integrated tile."""
    for value, length in zip((dummy, probe_x, probe_y),
                             (info.shape[0], info.shape[2], info.shape[3])):
        if not isinstance(value, (int, np.integer)) or not 0 <= value < length:
            raise ValueError("Probe or dummy index is outside the scan")
    cx, cy = info.chunks[2:4]
    spectra, totals, finite = cached_spectrum_tile(
        info, dummy, probe_x // cx, probe_y // cy, radius, offset_px, offset_py)
    x, y = probe_x % cx, probe_y % cy
    return spectrum_from_tile(spectra, totals, finite, x, y, normalize)


def spectrum_tile_tasks(infos):
    """List spatial tiles for the selected 6D Zarr scans."""
    grids = [(info, (info.shape[2] + info.chunks[2] - 1) // info.chunks[2],
               (info.shape[3] + info.chunks[3] - 1) // info.chunks[3])
             for info in infos if info.chunks is not None and len(info.shape) == 6]
    if sum(nx * ny for _, nx, ny in grids) > 256:
        return []  # Avoid evicting early tiles before preparation has finished.
    return [(info, x, y) for info, nx, ny in grids for x in range(nx) for y in range(ny)]


@st.cache_data(show_spinner=False, max_entries=128)
def cached_spectrum(info, dummy, probe_x, probe_y, radius, offset_px, offset_py, normalize):
    if info.chunks is not None and len(info.shape) == 6:
        return spectrum_from_zarr_tile(info, dummy, probe_x, probe_y, radius,
                                       offset_px, offset_py, normalize)
    return extract_spectrum(info, dummy=dummy, probe_x=probe_x, probe_y=probe_y,
                            radius=radius, offset_px=offset_px, offset_py=offset_py,
                            normalize_3d=normalize)


@st.cache_data(show_spinner=False, max_entries=256)
def _cached_inspect(path_str, mtime_ns, size, revision):
    return inspect_scan(path_str)


def inspect_scan_cached(path):
    """Invalidate Zarr caches when metadata or any chunk file changes."""
    return _cached_inspect(str(path), *scan_revision(path))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_export_csv(curves):
    from eels_studio.io.spectra_exports import export_csv
    return export_csv(curves)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_export_npz(curves, settings):
    from eels_studio.io.spectra_exports import export_npz
    return export_npz(curves, settings)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_notebook_npy(curves):
    from eels_studio.io.spectra_exports import export_notebook_npy
    return export_notebook_npy(curves)
