"""Run with: .venv/bin/python -m streamlit run app.py"""
from pathlib import Path
import io
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from eels_core import (
    circular_detector_mask, detector_center, diffraction_pattern, display_intensity,
    detector_offsets_from_click, energy_loss_axis_mev, export_csv, export_npz,
    extract_spectrum, gaussian_broaden_spectrum, inspect_scan, detailed_balance_factor,
    KB_MEV_PER_K,
)
from detector_click import register_detector_click_bridge
from angle_resolved import render_angle_resolved

ROOT = Path(__file__).resolve().parent
detector_click_bridge = register_detector_click_bridge()
COLORS = ["#137c78", "#dd7848", "#626cc6", "#c24c79", "#7a9845", "#428fbd"]
LINE_STYLES = {
    "Solid": ("solid", "-"),
    "Dashed": ("dash", "--"),
    "Dotted": ("dot", ":"),
    "Dash-dot": ("dashdot", "-."),
}
st.set_page_config(page_title="EELS Studio", page_icon="🔬", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 3.5rem; padding-bottom: 2rem;}
h1 {letter-spacing: -0.045em;}
[data-testid="stMetric"] {background: white; border: 1px solid #dce5ec;
  border-radius: 12px; padding: 14px 18px;}
[data-testid="stSidebar"] {border-right: 1px solid #dce5ec;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_spectrum(info, sample, probe_x, probe_y, radius, offset_px, offset_py, normalize):
    return extract_spectrum(info, sample=sample, probe_x=probe_x, probe_y=probe_y,
                            radius=radius, offset_px=offset_px, offset_py=offset_py,
                            normalize_3d=normalize)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_pattern(info, energy_index, sample, probe_x, probe_y):
    return diffraction_pattern(info, energy_index, sample=sample, probe_x=probe_x, probe_y=probe_y)


@st.cache_data(show_spinner=False, max_entries=8)
def figure_downloads(curves, mode, x_limits, y_limits, normalize):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for curve in curves:
        style = curve["style"]
        ax.plot(curve["energy"], display_intensity(curve["intensity"], mode),
                label=curve["label"], color=style["color"], linewidth=style["width"],
                linestyle=LINE_STYLES[style["line_style"]][1])
    ax.set_xlabel("Energy loss (meV)")
    ax.set_ylabel(intensity_label(mode, normalize))
    ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=0.16)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    result = {}
    for extension in ("svg", "png"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=300, bbox_inches="tight")
        result[extension] = buffer.getvalue()
    plt.close(fig)
    return result


def intensity_label(mode, normalize):
    base = "Normalized detector intensity" if normalize else "Detector intensity"
    return f"log10({base.lower()})" if mode == "log10" else base


def suggested_label(path):
    match = re.search(r"(?:^|_)T(\d+(?:\.\d+)?)K(?:_|$)", Path(path).stem)
    return f"{match[1]} K" if match else Path(path).stem


def suggested_temperature(path):
    match = re.search(r"(?:^|_)T(\d+(?:\.\d+)?)K(?:_|$)", Path(path).stem)
    return float(match[1]) if match and float(match[1]) > 0 else None


def index_control(label, count, key):
    # Keep widget state valid when the user switches to a smaller scan.
    if key in st.session_state and st.session_state[key] >= count:
        st.session_state[key] = 0
    return int(st.number_input(label, min_value=0, max_value=count - 1, value=0,
                               step=1, key=key, disabled=count == 1))


def move_detector_from_click():
    # Component callbacks run before widgets, so updating the sidebar is safe.
    clicked = st.session_state.get("detector_click", {}).get("clicked")
    context = st.session_state.get("detector_preview_context")
    if not clicked or not context or clicked.get("preview_id") != context["preview_id"]:
        return
    try:
        dx, dy = detector_offsets_from_click(context["shape"], clicked["px"], clicked["py"],
                                              context["selected_shapes"])
        st.session_state["offset_px"] = dx
        st.session_state["offset_py"] = dy
    except (ValueError, KeyError, TypeError) as exc:
        st.session_state["detector_click_error"] = str(exc)


def reset_detector_center():
    st.session_state["offset_px"] = 0
    st.session_state["offset_py"] = 0


st.caption("STEM · SPECTROSCOPY WORKSPACE")
st.title("EELS Studio")
st.write("From diffraction data to a spectrum. Choose your scans, position the detector, and compare.")

with st.sidebar:
    st.header("Your scans")
    folder = st.text_input("Data folder", value=str(ROOT), help="Local folder on the computer running this app.")
    st.button("Refresh files", use_container_width=True)
    try:
        directory = Path(folder).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("Enter an existing folder")
        available = sorted(directory.glob("*.npy"))
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    if not available:
        st.info("No .npy files in this folder. Enter a folder containing your scans.")
        st.stop()
    # Inspection reads only headers; arrays with unsupported dimensions are excluded.
    scans, rejected = {}, []
    for path in available:
        try:
            scans[str(path)] = inspect_scan(path)
        except (OSError, ValueError, EOFError) as exc:
            rejected.append(f"{path.name}: {exc}")
    if rejected:
        with st.expander(f"{len(rejected)} unsupported file(s)"):
            for reason in rejected:
                st.caption(reason)
    options = list(scans)
    if not options:
        st.info("No supported 3D or 6D scan arrays found.")
        st.stop()
    selected = st.multiselect("Files to compare", options, default=options[:2],
                             format_func=lambda p: Path(p).name, key=f"files:{directory}")
    if not selected:
        st.info("Select at least one scan to get started.")
        st.stop()
    infos = [scans[p] for p in selected]
    with st.expander("Curve labels", expanded=False):
        labels = [st.text_input(Path(p).name, value=suggested_label(p), key=f"label:{p}").strip()
                  for p in selected]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        st.warning("Give each selected file a nonempty, unique curve label.")
        st.stop()

    st.divider()
    st.header("Extraction")
    radius = st.number_input("Detector radius (pixels)", min_value=0.1, value=20.0, step=1.0)
    st.caption("Center offsets from (px // 2, py // 2). px is the vertical image axis; py is horizontal.")
    min_px = min(i.shape[-2] for i in infos)
    min_py = min(i.shape[-1] for i in infos)
    for key, length in (("offset_px", min_px), ("offset_py", min_py)):
        if key in st.session_state:
            clamped = max(-(length // 2), min((length - 1) // 2, st.session_state[key]))
            if clamped != st.session_state[key]:
                st.session_state[key] = clamped
    col1, col2 = st.columns(2)
    offset_px = col1.number_input("px offset", min_value=-(min_px // 2), max_value=(min_px - 1) // 2,
                                  value=0, step=1, key="offset_px")
    offset_py = col2.number_input("py offset", min_value=-(min_py // 2), max_value=(min_py - 1) // 2,
                                  value=0, step=1, key="offset_py")
    st.button("Reset detector center", on_click=reset_detector_center, use_container_width=True)
    if "detector_click_error" in st.session_state:
        st.warning(st.session_state.pop("detector_click_error"))
    normalize = st.checkbox("Normalize full probe block", value=True,
                            help="Divide by the sum over all energy bins and all detector-plane pixels at this probe, before detector integration. Matches normalize_3d in the notebook.")
    six_d = [i for i in infos if len(i.shape) == 6]
    if six_d:
        sample = index_control("Sample index", min(i.shape[0] for i in six_d), "sample")
        probe_y = index_control("Probe y index", min(i.shape[3] for i in six_d), "probe_y")
        x_options = list(range(min(i.shape[2] for i in six_d)))
        if "probe_x" in st.session_state:
            st.session_state["probe_x"] = [x for x in st.session_state["probe_x"] if x in x_options]
        probe_xs = st.multiselect("Probe x indices", x_options, default=[0], key="probe_x")
        if not probe_xs:
            st.info("Select at least one probe x index.")
            st.stop()
        if len(six_d) != len(infos):
            st.caption("3D files contribute one curve at sample 0, probe (0, 0). Index controls apply to 6D scans.")
    else:
        sample, probe_y, probe_xs = 0, 0, [0]
        st.caption("3D diffraction data · one spectrum per file.")

    with st.expander("Energy calibration", expanded=True):
        timestep = st.number_input("Simulation time step (fs)", min_value=0.000001,
                                   value=2.5, step=0.5, format="%.6f")
        stride = st.number_input("Sampling stride", min_value=1, value=3, step=1)
        st.caption("Confirm these values for your simulation. .npy arrays do not store time calibration. Applied to every selected file.")
        ordering = st.selectbox("Input energy ordering", ["FFT-shifted (notebook default)", "Unshifted FFT"])

    with st.expander("Detailed balance", expanded=True):
        apply_detailed_balance = st.checkbox("Apply detailed-balance factor", value=False,
                                             key="apply_detailed_balance")
        st.latex(r"f(E,T)=\frac{\beta E}{1-e^{-\beta E}},\qquad \beta=(k_B T)^{-1}")
        temperatures = {}
        st.caption("Set each scan's temperature in kelvin. Values inferred from T…K filenames are editable. Positive energy means loss; f(0,T) = 1. Applied before Gaussian broadening and log display.")
        for info, label in zip(infos, labels):
            temperature = st.number_input(
                f"Temperature (K) · {label}", min_value=0.000001,
                value=suggested_temperature(info.path), step=1.0, format="%.6f",
                key=f"temperature:{info.path}", placeholder="Enter temperature in K",
                disabled=not apply_detailed_balance)
            if apply_detailed_balance:
                temperatures[info.path] = temperature
        if apply_detailed_balance and any(t is None for t in temperatures.values()):
            st.info("Enter a positive temperature for every selected scan to apply detailed balance.")
            st.stop()

    with st.expander("Gaussian broadening", expanded=True):
        broaden = st.checkbox("Broaden EELS spectrum", value=False)
        sigma_input = st.number_input("Gaussian σ (meV)", min_value=0.0, value=1.0, step=0.5,
                                      disabled=not broaden, help="Standard deviation of the Gaussian, applied to linear intensities after detector integration.")
        sigma_mev = sigma_input if broaden else 0.0
        if broaden:
            st.caption(f"FWHM = {sigma_mev * np.sqrt(8 * np.log(2)):.3f} meV. Applies to spectra and downloads; the diffraction image stays unchanged.")

settings = dict(detector_radius_px=radius, center_offset_px=offset_px, center_offset_py=offset_py,
                normalize_3d=normalize, sample=sample, probe_x_indices=probe_xs,
                probe_y=probe_y, timestep_fs=timestep, stride=stride, input_energy_ordering=ordering,
                gaussian_sigma_mev=sigma_mev, gaussian_boundary="reflect", gaussian_truncate=4.0,
                detailed_balance_enabled=apply_detailed_balance,
                detailed_balance_temperatures_k=temperatures,
                detailed_balance_formula="beta*E / (1 - exp(-beta*E)); f(0)=1",
                boltzmann_constant_mev_per_k=KB_MEV_PER_K,
                processing_order=["detector integration / optional full-probe normalization",
                                  "energy ordering", "optional detailed balance",
                                  "optional Gaussian broadening", "display transform"])
curves = []
try:
    with st.spinner("Integrating detector intensities…"):
        for info, label in zip(infos, labels):
            xs = probe_xs if len(info.shape) == 6 else [0]
            s, y = (sample, probe_y) if len(info.shape) == 6 else (0, 0)
            energy = energy_loss_axis_mev(info.canonical_shape[1], timestep, stride)
            factor = detailed_balance_factor(energy, temperatures[info.path]) if apply_detailed_balance else None
            for x in xs:
                intensity = cached_spectrum(info, s, x, y, radius, offset_px, offset_py, normalize)
                if ordering == "Unshifted FFT":
                    intensity = np.fft.fftshift(intensity)
                if factor is not None:
                    intensity = intensity * factor
                if sigma_mev > 0:
                    intensity = gaussian_broaden_spectrum(energy, intensity, sigma_mev)
                name = f"{label} · x={x}, y={y}" if len(info.shape) == 6 else label
                curves.append(dict(label=name, path=info.path, sample=s, probe_x=x, probe_y=y,
                                   energy=energy, intensity=intensity,
                                   detailed_balance_enabled=apply_detailed_balance,
                                   temperature_k=temperatures.get(info.path)))
except (OSError, ValueError, IndexError) as exc:
    st.error(f"Could not extract spectra: {exc}")
    st.stop()

metrics = st.columns(4)
metrics[0].metric("Selected scans", len(infos))
metrics[1].metric("Spectra", len(curves))
bins = sorted({len(c["energy"]) for c in curves})
metrics[2].metric("Energy bins", " / ".join(map(str, bins)))
resolutions = sorted({round(float(c["energy"][1] - c["energy"][0]), 6) for c in curves if len(c["energy"]) > 1})
metrics[3].metric("Resolution (meV)", " / ".join(f"{v:.3f}" for v in resolutions) or "—")

spectrum_tab, detector_tab, angle_tab, details_tab = st.tabs(
    ["Spectra", "Detector preview", "Angle-resolved EELS", "Files & method"])
with spectrum_tab:
    controls = st.columns([1.3, 1, 1])
    mode_label = controls[0].selectbox("Intensity display", ["log10", "Linear"])
    mode = "log10" if mode_label == "log10" else "linear"
    x_min = controls[1].number_input("Energy min (meV)", value=-150.0, step=10.0)
    x_max = controls[2].number_input("Energy max (meV)", value=150.0, step=10.0)
    if x_min >= x_max:
        st.error("Energy min must be less than energy max.")
        st.stop()
    x_limits = (x_min, x_max)
    with st.expander("Line appearance", expanded=False):
        st.caption("Choose a spectrum to customize. Styles also apply to SVG/PNG downloads and are saved in NumPy metadata.")
        # Keep styles separate from widget state so removing/reselecting a curve
        # or editing its label does not discard its appearance during this session.
        saved_styles = st.session_state.setdefault("curve_styles", {})
        curve_keys = [json.dumps([c["path"], c["sample"], c["probe_x"], c["probe_y"]]) for c in curves]
        curve_labels = dict(zip(curve_keys, [c["label"] for c in curves]))
        for i, curve_key in enumerate(curve_keys):
            saved_styles.setdefault(curve_key, dict(color=COLORS[i % len(COLORS)], line_style="Solid", width=2.0))
        if "style_editor" in st.session_state and st.session_state["style_editor"] not in curve_keys:
            st.session_state["style_editor"] = curve_keys[0]
        editing = st.selectbox("Spectrum to style", curve_keys, format_func=curve_labels.get, key="style_editor")
        current = saved_styles[editing]
        color_col, style_col, width_col = st.columns(3)
        color = color_col.color_picker("Line color", value=current["color"], key=f"line_color:{editing}")
        line_style = style_col.selectbox("Line style", list(LINE_STYLES),
                                         index=list(LINE_STYLES).index(current["line_style"]),
                                         key=f"line_style:{editing}")
        line_width = width_col.number_input("Line width", min_value=0.5, max_value=8.0,
                                            value=current["width"], step=0.5, key=f"line_width:{editing}")
        saved_styles[editing] = dict(color=color, line_style=line_style, width=line_width)
        for curve, curve_key in zip(curves, curve_keys):
            curve["style"] = saved_styles[curve_key].copy()
    y_limits = None
    with st.expander("Vertical plot limits"):
        if st.checkbox("Set intensity limits"):
            left, right = st.columns(2)
            y_min = left.number_input("Intensity min (display units)", value=-10.0)
            y_max = right.number_input("Intensity max (display units)", value=0.0)
            if y_min >= y_max:
                st.error("Intensity min must be less than intensity max.")
                st.stop()
            y_limits = (y_min, y_max)
    show_hover = st.toggle("Show hover details", value=True, key="show_hover_details",
                           help="Show or hide the energy and intensity popup when hovering over the spectra.")
    fig = go.Figure()
    for curve in curves:
        style = curve["style"]
        fig.add_trace(go.Scatter(x=curve["energy"], y=display_intensity(curve["intensity"], mode),
                                 name=curve["label"], mode="lines",
                                 line=dict(color=style["color"], width=style["width"],
                                           dash=LINE_STYLES[style["line_style"]][0]),
                                 hovertemplate="%{y:.6g}<extra>%{fullData.name}</extra>"))
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=25, b=20), template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified" if show_hover else False,
                      legend=dict(orientation="h", y=-0.2, x=0),
                      xaxis=dict(title="Energy loss (meV)", range=x_limits, zerolinecolor="#bdcbd4",
                                 hoverformat=".4f"),
                      yaxis=dict(title=intensity_label(mode, normalize), range=y_limits))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False,
                    "toImageButtonOptions": {"format": "svg", "filename": "eels_spectra"}})
    st.caption("Drag to zoom · double-click to reset · click a legend entry to hide a curve. Downloads include every selected curve.")
    if apply_detailed_balance:
        st.caption("Detailed-balance correction active: " + "; ".join(
            f"{label}: {temperatures[info.path]:g} K" for info, label in zip(infos, labels)))
    if sigma_mev > 0:
        st.caption(f"Gaussian broadening active: σ = {sigma_mev:g} meV (FWHM = {sigma_mev * np.sqrt(8 * np.log(2)):.3f} meV).")
    if mode == "log10" and any(np.any(c["intensity"] <= 0) for c in curves):
        st.caption("Nonpositive values are clipped to the smallest positive float for log10 display, matching the notebook. Exported data keeps the original values.")
    if not any(np.any((c["energy"] >= x_min) & (c["energy"] <= x_max)) for c in curves):
        st.warning("The selected energy window contains no data bins.")
    settings["plot"] = dict(mode=mode, x_limits=x_limits, y_limits=y_limits, show_hover_details=show_hover)
    with st.expander("Export spectra & figures", expanded=True):
        st.caption("Data exports contain the full energy range and linear intensities. Figures use the display controls above; browser-only zoom and legend changes are not applied.")
        if apply_detailed_balance:
            st.caption("All downloads include the detailed-balance factor at each scan's temperature. NPZ records the temperatures and correction settings.")
        if sigma_mev > 0:
            st.caption(f"All downloads include Gaussian broadening (σ = {sigma_mev:g} meV). Turn broadening off to export unbroadened spectra; NPZ records the processing settings.")
        exports = st.columns(5)
        exports[0].download_button("CSV data", export_csv(curves), "eels_spectra.csv", "text/csv", use_container_width=True)
        exports[1].download_button("NumPy + settings", export_npz(curves, settings), "eels_spectra.npz", "application/octet-stream", use_container_width=True)
        if len(bins) == 1:
            buffer = io.BytesIO()
            np.save(buffer, np.stack([np.column_stack((c["energy"], c["intensity"])) for c in curves]))
            exports[2].download_button("Notebook .npy", buffer.getvalue(), "eels_spectra.npy", "application/octet-stream", use_container_width=True)
        else:
            exports[2].caption("Use NPZ for unequal energy lengths.")
        figures = figure_downloads(curves, mode, x_limits, y_limits, normalize)
        exports[3].download_button("SVG figure", figures["svg"], "eels_spectra.svg", "image/svg+xml", use_container_width=True)
        exports[4].download_button("PNG figure", figures["png"], "eels_spectra.png", "image/png", use_container_width=True)

with detector_tab:
    preview_cols = st.columns([2, 1, 1])
    preview_index = preview_cols[0].selectbox("Preview spectrum", list(range(len(curves))),
                                              format_func=lambda i: curves[i]["label"])
    requested_energy = preview_cols[1].number_input("Preview energy (meV)", value=0.0, step=1.0)
    pattern_log = preview_cols[2].checkbox("Log diffraction image", value=True)
    curve = curves[preview_index]
    info = scans[curve["path"]]
    axis = curve["energy"]
    energy_index = int(np.argmin(np.abs(axis - requested_energy)))
    raw_index = int(np.fft.fftshift(np.arange(len(axis)))[energy_index]) if ordering == "Unshifted FFT" else energy_index
    try:
        pattern = cached_pattern(info, raw_index, curve["sample"], curve["probe_x"], curve["probe_y"])
        if not np.isfinite(pattern).all():
            raise ValueError("Diffraction plane contains NaN or infinite intensities")
        if pattern_log:
            positive = pattern[pattern > 0]
            if not positive.size:
                raise ValueError("This plane has no positive intensities. Turn off log diffraction image.")
            shown = np.log10(np.clip(pattern, positive.min(), None))
        else:
            shown = pattern
        center = detector_center(info, offset_px, offset_py)
        mask = circular_detector_mask(pattern.shape, center, radius)
        image = go.Figure(go.Heatmap(z=shown, colorscale="Viridis",
                                     colorbar=dict(title="log10(I)" if pattern_log else "Intensity"),
                                     hovertemplate="py=%{x}<br>px=%{y}<br>%{z:.5g}<extra></extra>"))
        image.add_shape(type="circle", x0=center[1] - radius, x1=center[1] + radius,
                        y0=center[0] - radius, y1=center[0] + radius,
                        line=dict(color="#ff765e", width=2))
        # Draw the center as shapes so it cannot steal clicks from nearby pixels.
        image.add_shape(type="line", x0=center[1] - 2, x1=center[1] + 2,
                        y0=center[0], y1=center[0], line=dict(color="#ff765e", width=2))
        image.add_shape(type="line", x0=center[1], x1=center[1],
                        y0=center[0] - 2, y1=center[0] + 2, line=dict(color="#ff765e", width=2))
        image.update_layout(height=570, template="plotly_white", margin=dict(l=20, r=20, t=20, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(title="py (pixel)", range=[-0.5, pattern.shape[1] - 0.5], constrain="domain"),
                            yaxis=dict(title="px (pixel)", range=[-0.5, pattern.shape[0] - 0.5], scaleanchor="x", scaleratio=1))
        st.caption("Click the diffraction image to position the detector. Drag to zoom; double-click to reset zoom. The same center offsets apply to all selected scans.")
        with st.container(key="detector_click_target"):
            st.plotly_chart(image, key="detector_preview", use_container_width=True, config={"displaylogo": False})
        preview_id = json.dumps([info.path, info.mtime_ns, curve["sample"], curve["probe_x"], curve["probe_y"], raw_index])
        st.session_state["detector_preview_context"] = dict(preview_id=preview_id, shape=pattern.shape,
                                                           selected_shapes=[i.shape[-2:] for i in infos])
        # Renew the bridge after each Streamlit redraw, including manual edits.
        st.session_state["detector_click_revision"] = st.session_state.get("detector_click_revision", 0) + 1
        detector_click_bridge(key="detector_click", data={"preview_id": preview_id,
                              "revision": st.session_state["detector_click_revision"]},
                              on_clicked_change=move_detector_from_click, height=0)
        st.caption(f"Nearest energy bin: {axis[energy_index]:.6g} meV · array index {raw_index} · "
                   f"center (px, py) = {center} · {int(mask.sum()):,} detector pixels. Image shows raw plane intensities.")
        if any(c - radius < 0 or c + radius > n - 1 for c, n in zip(center, pattern.shape)):
            st.warning("The detector extends beyond this array. Only pixels inside the recorded plane are integrated.")
    except (OSError, ValueError, IndexError) as exc:
        st.warning(f"Preview unavailable: {exc}")

with angle_tab:
    render_angle_resolved(curves, scans, settings)

with details_tab:
    st.dataframe([{"Label": label, "File": Path(info.path).name, "Shape": str(info.shape),
                   "Type": info.dtype, "Size (MB)": round(info.size / 1e6, 1)}
                  for info, label in zip(infos, labels)], use_container_width=True, hide_index=True)
    st.markdown("""
**Array conventions**

- 3D: `(energy, px, py)`, one probe block per file.
- 6D: `(sample, energy, probe_x, probe_y, px, py)`, selected sample and probe positions.
- The circular detector uses `(px − center_px)² + (py − center_py)² ≤ radius²`.
- Full-probe normalization divides by the sum over **all energy bins and all pixels** at that probe.
- Energy is `fftshift(fftfreq(n_energy, timestep_fs × stride / 1000)) × 4.13566769692386` in meV.
- Input intensities are assumed FFT-shifted by default, as in the notebook. Choose unshifted only if your simulation output uses raw FFT ordering.
- Optional detailed balance multiplies linear intensity by `βE / (1 − exp(−βE))`, with `β = 1/(k_B T)`, energy in meV, and a separate temperature in kelvin for each file. Positive energy means loss; the zero-energy factor is 1. It is applied after energy ordering and before Gaussian broadening, without renormalizing the corrected spectrum. Enable this only for spectra that still need this correction.
- Optional Gaussian broadening operates on linear EELS intensity after detector integration and before log display. σ is entered in meV and converted to bins using each file's energy spacing. The kernel extends to 4σ; reflecting boundaries preserve the recorded sum without wrapping between energy endpoints. Edge features can be affected by this boundary assumption.

Files are memory-mapped and integrated in small energy blocks. Changing the plot or preview reuses cached spectra.
Different energy lengths are supported; shared time step and stride must be appropriate for every selected scan.
""")
    st.code(json.dumps(settings, indent=2), language="json")
