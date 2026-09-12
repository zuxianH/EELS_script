"""Interactive raw detector map at selected energies of a 6D probe scan."""
import io
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from eels_core import circular_detector_mask, detector_center, detector_scan_map, energy_loss_axis_mev
from cache_layer import PNG_DPI_OPTIONS


@st.cache_data(show_spinner=False, max_entries=64)
def cached_scan_map(info, raw_index, sample, radius, offset_px, offset_py):
    return detector_scan_map(info, raw_index, sample=sample, radius=radius,
                             offset_px=offset_px, offset_py=offset_py)


def parse_map_energies(text):
    """Parse finite meV values separated by commas, semicolons, or whitespace."""
    tokens = [token for token in re.split(r"[,;\s]+", text.strip()) if token]
    if not tokens:
        raise ValueError("Enter at least one map energy, for example: 20, 40, 60")
    try:
        energies = [float(token) for token in tokens]
    except ValueError:
        raise ValueError("Enter numeric map energies separated by commas, for example: 20, 40, 60") from None
    if not np.isfinite(energies).all():
        raise ValueError("Map energies must be finite numbers")
    return energies


def selected_map_bins(axis, energies, ordering):
    """Group requests mapping to the same stored bin, preserving input order."""
    raw_indices = np.arange(len(axis))
    if ordering == "Unshifted FFT":
        raw_indices = np.fft.fftshift(raw_indices)
    bins = {}
    for requested in energies:
        index = int(np.argmin(np.abs(axis - requested)))
        raw_index = int(raw_indices[index])
        entry = bins.setdefault(raw_index, dict(
            energy_index=raw_index, selected_energy_mev=float(axis[index]), requested_energies_mev=[]))
        entry["requested_energies_mev"].append(requested)
    return list(bins.values())


@st.cache_data(show_spinner=False, max_entries=8)
def maps_png(scan_maps, titles, columns=3, color_limits=None, dpi=300):
    columns = min(columns, len(scan_maps))
    rows = (len(scan_maps) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 6 * rows),
                             layout="constrained", squeeze=False)
    try:
        used_axes = list(axes.flat)[:len(scan_maps)]
        image = None
        for ax, scan_map, title in zip(used_axes, scan_maps, titles):
            image = ax.imshow(scan_map.T, origin="lower", cmap="viridis", interpolation="nearest",
                               vmin=color_limits[0] if color_limits else None,
                               vmax=color_limits[1] if color_limits else None)
            ax.set(xlabel="Probe x (index)", ylabel="Probe y (index)", title=title)
            # Sharing one scale makes a colorbar per panel redundant; show one for all.
            if color_limits is None:
                fig.colorbar(image, ax=ax, label="Detector-summed intensity (a.u.)")
        for ax in list(axes.flat)[len(scan_maps):]:
            ax.set_visible(False)
        if color_limits is not None:
            fig.colorbar(image, ax=used_axes, label="Detector-summed intensity (a.u.)", shrink=0.8)
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=dpi, bbox_inches="tight")
        return output.getvalue()
    finally:
        plt.close(fig)


def map_png(scan_map, title, color_limits=None, dpi=300):
    return maps_png([scan_map], [title], columns=1, color_limits=color_limits, dpi=dpi)


@st.cache_data(show_spinner=False, max_entries=64)
def map_npy_bytes(scan_map):
    buffer = io.BytesIO()
    np.save(buffer, scan_map)
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=64)
def map_npz_bytes(scan_map, metadata):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, scan_map=scan_map, metadata_json=np.array(json.dumps(metadata)))
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=8)
def all_maps_npz_bytes(scan_maps, bins, base_metadata):
    metadata = dict(base_metadata, axes=["energy_map", "probe_x", "probe_y"], maps=bins)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, scan_maps=np.stack(scan_maps),
                        selected_energies_mev=np.array([entry["selected_energy_mev"] for entry in bins]),
                        energy_indices=np.array([entry["energy_index"] for entry in bins]),
                        metadata_json=np.array(json.dumps(metadata)))
    return buffer.getvalue()



def render_scan_map(infos, labels):
    six_d = {info.path: (info, label) for info, label in zip(infos, labels) if len(info.shape) == 6}
    if not six_d:
        st.info("Select a 6D scan to visualize a 2D probe map. 3D arrays contain only one probe position.")
        return
    with st.sidebar:
        st.divider()
        st.header("2D scan map")
        if st.session_state.get("map_scan") not in six_d:
            st.session_state["map_scan"] = next(iter(six_d))
        path = st.selectbox("Map scan", list(six_d), format_func=lambda p: six_d[p][1], key="map_scan")
        info, _ = six_d[path]
        sample = 0
        energy_text = st.text_input("Map energies (meV)", value="60", key="map_energies",
                                    help="Enter one or more energies separated by commas, for example: 20, 40, 60, 80.")
        shared_scale = st.checkbox("Shared color scale", value=True, key="map_shared_scale",
                                   help="Use the same intensity range across all maps. Turn off to show each map's spatial contrast on its own scale.")
        columns_per_row = st.number_input("Maps per row", min_value=1, max_value=12, value=3, step=1,
                                          key="map_columns",
                                          help="Grid layout for the panels below and the combined PNG export, e.g. 2 for a 2-wide grid, 1 to stack maps in a single column.")
        radius = st.number_input("Detector radius (pixels)", min_value=0.1, value=21.0,
                                 step=1.0, key="map_radius")
        st.caption("Center offsets from (px // 2, py // 2), in detector pixels.")
        offsets = []
        for axis, size in zip(("px", "py"), info.shape[-2:]):
            key = f"map_offset_{axis}"
            lo, hi = -(size // 2), (size - 1) // 2
            if key in st.session_state:
                st.session_state[key] = max(lo, min(hi, st.session_state[key]))
            offsets.append(st.number_input(f"{axis} offset", min_value=lo, max_value=hi,
                                           value=0, step=1, key=key))
        timestep = st.number_input("Simulation time step (fs)", min_value=0.000001,
                                   value=5.0, step=0.5, format="%.6f", key="map_timestep")
        stride = st.number_input("Sampling stride", min_value=1, value=3, step=1, key="map_stride")
        ordering = st.selectbox("Input energy ordering", ["FFT-shifted (notebook default)", "Unshifted FFT"],
                                key="map_ordering")
        st.caption("Confirm the time calibration for this simulation. The map covers every probe x and y.")
    st.subheader("2D scan maps")
    st.caption("Raw detector-summed intensity at each selected energy bin, shown on a linear color scale. "
               "Map controls are independent of spectrum controls; normalization and broadening "
               "do not apply to these maps.")
    try:
        requested_energies = parse_map_energies(energy_text)
        axis = energy_loss_axis_mev(info.shape[1], timestep, stride)
        bins = selected_map_bins(axis, requested_energies, ordering)
        center = detector_center(info, *offsets)
        mask = circular_detector_mask(info.shape[-2:], center, radius)
        with st.spinner("Summing the detector across probe rows at selected energies…"):
            # Each read reduces one energy slice to a small map before the next
            # energy is read. Only these reduced maps are kept in memory.
            scan_maps = [cached_scan_map(info, entry["energy_index"], sample, radius, *offsets)
                         for entry in bins]
        limits = ((min(float(m.min()) for m in scan_maps), max(float(m.max()) for m in scan_maps))
                  if shared_scale else None)
        metrics = st.columns(3)
        metrics[0].metric("Energy maps", len(bins))
        metrics[1].metric("Probe positions", str(scan_maps[0].size))
        metrics[2].metric("Detector pixels", str(int(mask.sum())))
        if len(bins) < len(requested_energies):
            st.info("Some requested energies select the same recorded bin; each distinct bin is shown once.")
        st.caption(f"Detector center (px, py) = {center}. Horizontal = probe x; vertical = probe y. "
                   + ("Shared intensity range across all maps." if shared_scale else "Each map uses its own intensity range."))
        png_dpi = st.selectbox("PNG export resolution", PNG_DPI_OPTIONS, index=PNG_DPI_OPTIONS.index(300),
                               format_func=lambda d: f"{d} DPI", key="png_dpi_scan_map",
                               help="Resolution used for the PNG downloads below.")
        base_metadata = dict(source_file=info.path, sample=sample,
                             timestep_fs=timestep, stride=stride, input_energy_ordering=ordering,
                             detector_radius_px=radius, detector_center_px_py=center,
                             axes=["probe_x", "probe_y"], intensity="raw detector sum",
                             shared_color_scale=shared_scale, color_limits=limits)
        titles = [f"{entry['selected_energy_mev']:.6g} meV" for entry in bins]
        stem = f"{Path(info.path).stem}_scan_map"
        n_columns = min(columns_per_row, len(bins))
        for start in range(0, len(bins), n_columns):
            panels = st.columns(n_columns)
            for panel, i in zip(panels, range(start, min(start + n_columns, len(bins)))):
                with panel:
                    entry, scan_map, title = bins[i], scan_maps[i], titles[i]
                    raw_index = entry["energy_index"]
                    if np.all(scan_map == 0):
                        st.info("This detector-integrated map is all zero; no spatial contrast is present in this slice.")
                    figure = go.Figure(go.Heatmap(
                        x=np.arange(scan_map.shape[0]), y=np.arange(scan_map.shape[1]), z=scan_map.T,
                        colorscale="Viridis", zsmooth=False, colorbar=dict(title="Intensity (a.u.)"),
                        zmin=limits[0] if limits else None, zmax=limits[1] if limits else None,
                        hovertemplate="probe x=%{x}<br>probe y=%{y}<br>Intensity=%{z:.6g}<extra></extra>"))
                    figure.update_layout(
                        title=title, height=650 if len(bins) == 1 else 450,
                        template="plotly_white", margin=dict(l=30, r=20, t=80, b=40),
                        xaxis=dict(title="Probe x (index)", range=[-0.5, scan_map.shape[0] - 0.5], constrain="domain"),
                        yaxis=dict(title="Probe y (index)", range=[-0.5, scan_map.shape[1] - 0.5], scaleanchor="x", scaleratio=1))
                    panel_stem = f"{stem}_E{raw_index}"
                    st.plotly_chart(figure, key=f"scan_map_plot:{raw_index}", use_container_width=True,
                                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": panel_stem}})
                    metadata = dict(base_metadata, **entry)
                    st.download_button("Map .npy", map_npy_bytes(scan_map), panel_stem + ".npy", "application/octet-stream", key=f"map_npy:{raw_index}")
                    st.download_button("Map + settings .npz", map_npz_bytes(scan_map, metadata), panel_stem + ".npz", "application/octet-stream", key=f"map_npz:{raw_index}")
                    st.download_button("Map PNG", map_png(scan_map, title, limits, png_dpi), panel_stem + ".png", "image/png", key=f"map_png:{raw_index}")
        if len(bins) > 1:
            st.subheader("Download all maps")
            exports = st.columns(2)
            exports[0].download_button("All maps + settings .npz", all_maps_npz_bytes(scan_maps, bins, base_metadata),
                                       stem + "s.npz", "application/octet-stream", key="all_maps_npz")
            exports[1].download_button("All maps PNG", maps_png(scan_maps, titles, n_columns, limits, png_dpi),
                                       stem + "s.png", "image/png", key="all_maps_png")
    except (OSError, ValueError, IndexError, EOFError) as exc:
        st.error(f"Could not create scan maps: {exc}")
