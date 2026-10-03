"""Spectrum extraction controls, detector interaction, analysis tabs, and downloads."""
from pathlib import Path
import json

import numpy as np
import streamlit as st

from eels_studio.ui.plotting import go, COLORS, LINE_STYLES, figure_downloads, intensity_label, config_caption
from eels_studio.core.identity import curve_identity_key
from eels_studio.core.display import spectrum_display_intensity
from eels_studio.core.spectra import (
    detector_center, energy_loss_axis_mev, nearest_energy_index, process_spectrum,
)
from eels_studio.ui.cache_layer import (
    cached_spectrum, cached_diffraction_pattern, cached_spectrum_tile, spectrum_tile_tasks,
    cached_export_csv, cached_export_npz, cached_notebook_npy, PNG_DPI_OPTIONS,
)
from eels_studio.ui.components.detector_click import register_detector_click_bridge, move_detector_from_click, reset_detector_center
from eels_studio.ui.components.axis_scaling import register_axis_scaling
from eels_studio.ui.components.panel_layout import (
    ADJUSTABLE_PANELS_ENABLED, register_panel_layout, layout_settings, update_panel_layout, reset_panel_layout,
)
from eels_studio.ui.state import BackgroundState, input_fingerprints
from eels_studio.io import background_exports
from eels_studio.ui.background_view import render_background
from eels_studio.ui.angle_resolved import render_angle_resolved
from eels_studio.ui.workspace_config import render_config_download, restore_background


def render_spectra(infos, labels, scans, config_slot):
    # Registration belongs to the active runtime (including independent AppTests).
    detector_click_bridge = register_detector_click_bridge()
    axis_scaling = register_axis_scaling()
    panel_layout = register_panel_layout() if ADJUSTABLE_PANELS_ENABLED else None
    with st.sidebar:
        st.divider()
        st.header("Extraction")
        st.caption("Detector radius and center (px, py, radius) are now set directly below the "
                   "detector image in the Spectra tab.")
        min_px = min(i.shape[-2] for i in infos)
        min_py = min(i.shape[-1] for i in infos)
        for key, length in (("offset_px", min_px), ("offset_py", min_py)):
            if key in st.session_state:
                clamped = max(-(length // 2), min((length - 1) // 2, st.session_state[key]))
                if clamped != st.session_state[key]:
                    st.session_state[key] = clamped
        # These reflect the editable px/py/radius fields below the detector image: Streamlit
        # already applies any pending edit to session_state before this script runs, so reading
        # it here (ahead of those widgets' own call site) still sees this render's live value.
        radius = st.session_state.get("detector_radius", 20.0)
        offset_px = st.session_state.get("offset_px", 0)
        offset_py = st.session_state.get("offset_py", 0)
        if "detector_click_error" in st.session_state:
            st.warning(st.session_state.pop("detector_click_error"))
        normalize = st.checkbox("Normalize full probe block", value=True, key="normalize_probe",
                                help="Divide by the sum over all energy bins and all detector-plane pixels at this probe, before detector integration. Matches normalize_3d in the notebook.")
        six_d = [i for i in infos if len(i.shape) == 6]
        dummy = 0
        if six_d:
            n_x = min(i.shape[2] for i in six_d)
            n_y = min(i.shape[3] for i in six_d)
            position_options = [(x, y) for x in range(n_x) for y in range(n_y)]
            if "probe_positions" in st.session_state:
                valid_positions = set(position_options)
                st.session_state["probe_positions"] = [
                    tuple(position) for position in st.session_state["probe_positions"]
                    if tuple(position) in valid_positions
                ]
            probe_positions = st.multiselect(
                "Probe positions (x, y)", position_options, default=[(0, 0)],
                format_func=lambda position: f"({position[0]}, {position[1]})",
                key="probe_positions",
                help="Choose individual probe positions. Each selected pair produces one spectrum per 6D file.")
            if not probe_positions:
                st.info("Select at least one probe position (x, y).")
                render_config_download(config_slot)
                st.stop()
            if len(six_d) != len(infos):
                st.caption("3D files contribute one curve at probe (0, 0). Selected pairs apply to 6D scans.")
        else:
            probe_positions = [(0, 0)]
            st.caption("3D diffraction data · one spectrum per file.")

        tile_tasks = spectrum_tile_tasks(infos)
        if tile_tasks and len(tile_tasks) <= 256:
            if st.button("Prepare all Zarr probe spectra",
                         help="Read all selected 6D Zarr scans once for the current detector. "
                              "Afterwards, any probe position can reuse the prepared spectra. "
                              "Large scans can take several minutes. Changing detector geometry "
                              "or restarting the app requires preparation again."):
                progress = st.progress(0.0, text="Preparing probe spectra…")
                try:
                    for index, (info, tx, ty) in enumerate(tile_tasks, 1):
                        cached_spectrum_tile(info, dummy, tx, ty, radius, offset_px, offset_py)
                        progress.progress(index / len(tile_tasks),
                                          text=f"Prepared {index}/{len(tile_tasks)} probe groups")
                    st.success("All Zarr probe spectra prepared for this detector.")
                except (OSError, ValueError, IndexError, EOFError) as exc:
                    st.error(f"Could not prepare all spectra: {exc}")
                finally:
                    progress.empty()

        with st.expander("Energy calibration", expanded=True):
            timestep = st.number_input("Simulation time step (fs)", min_value=0.000001,
                                       value=2.5, step=0.5, format="%.6f", key="spectrum_timestep")
            stride = st.number_input("Sampling stride", min_value=1, value=3, step=1, key="spectrum_stride")
            st.caption("Confirm these values for your simulation. Time calibration is not inferred from NumPy or Zarr input. Applied to every selected scan.")
            ordering = st.selectbox("Input energy ordering", ["FFT-shifted (notebook default)", "Unshifted FFT"], key="spectrum_ordering")

        with st.expander("Gaussian broadening", expanded=True):
            broaden = st.checkbox("Broaden EELS spectrum", value=False, key="broaden_spectrum")
            sigma_input = st.number_input("Gaussian σ (meV)", min_value=0.0, value=1.0, step=0.5,
                                          disabled=not broaden, key="spectrum_sigma", help="Standard deviation of the Gaussian, applied to linear intensities after detector integration.")
            sigma_mev = sigma_input if broaden else 0.0
            if broaden:
                st.caption(f"FWHM = {sigma_mev * np.sqrt(8 * np.log(2)):.3f} meV. Applies to spectra and downloads; the diffraction image stays unchanged.")

    settings = dict(detector_radius_px=radius, center_offset_px=offset_px, center_offset_py=offset_py,
                    normalize_3d=normalize, dummy=dummy, probe_positions_xy=probe_positions,
                    timestep_fs=timestep, stride=stride, input_energy_ordering=ordering,
                    gaussian_sigma_mev=sigma_mev, gaussian_boundary="reflect", gaussian_truncate=4.0,
                    processing_order=["detector integration / optional full-probe normalization",
                                      "energy ordering", "optional Gaussian broadening",
                                      "display transform"])
    curves = []
    try:
        with st.spinner("Integrating detector intensities…"):
            for info, label in zip(infos, labels):
                positions = probe_positions if len(info.shape) == 6 else [(0, 0)]
                s = dummy
                energy = energy_loss_axis_mev(info.canonical_shape[1], timestep, stride)
                for x, y in positions:
                    intensity = cached_spectrum(info, s, x, y, radius, offset_px, offset_py, normalize)
                    intensity = process_spectrum(
                        energy, intensity, unshifted=ordering == "Unshifted FFT", sigma_mev=sigma_mev,
                    )
                    name = f"{label} · x={x}, y={y}" if len(info.shape) == 6 else label
                    curves.append(dict(label=name, path=info.path, dummy=s, probe_x=x, probe_y=y,
                                       energy=energy, intensity=intensity))
    except (OSError, ValueError, IndexError, EOFError) as exc:
        st.error(f"Could not extract spectra: {exc}")
        render_config_download(config_slot)
        st.stop()

    st.session_state["_comparison_curve_paths"] = [c["path"] for c in curves]

    with st.sidebar:
        with st.expander("Detector preview", expanded=True):
            if st.session_state.get("preview_index", 0) >= len(curves):
                st.session_state["preview_index"] = 0
            preview_index = st.selectbox("Preview spectrum", list(range(len(curves))),
                                         format_func=lambda i: curves[i]["label"], key="preview_index")
            requested_energy = st.number_input("Preview energy (meV)", value=0.0, step=1.0, key="preview_energy")
            pattern_log = st.checkbox("Log diffraction image", value=True, key="pattern_log")

    metrics = st.columns(4)
    metrics[0].metric("Selected scans", len(infos))
    metrics[1].metric("Spectra", len(curves))
    bins = sorted({len(c["energy"]) for c in curves})
    metrics[2].metric("Energy bins", " / ".join(map(str, bins)))
    resolutions = sorted({round(float(c["energy"][1] - c["energy"][0]), 6) for c in curves if len(c["energy"]) > 1})
    metrics[3].metric("Resolution (meV)", " / ".join(f"{v:.3f}" for v in resolutions) or "—")

    background_state = st.session_state.setdefault("background_state", BackgroundState())
    revisions = {info.path: (info.mtime_ns, info.size, info.revision) for info in infos}
    background_state.source_revisions = revisions
    had_background = bool(background_state.applied_results)
    if background_state.sync_inputs(input_fingerprints(curves, settings, revisions)):
        st.session_state["bg_signal"] = "Input"
        if had_background:
            st.info("Background results cleared because inputs or selected curves changed. Settings are retained for refitting.")
    restore_background(curves, background_state)
    if "bg_next_signal" in st.session_state:
        st.session_state["bg_signal"] = st.session_state.pop("bg_next_signal")
    spectrum_tab, background_tab, angle_tab, details_tab = st.tabs(
        ["Spectra", "Background", "Angle-resolved EELS", "Files & method"])
    with spectrum_tab:
        if background_state.applied_results:
            signal = st.radio("Spectrum signal", ["Input", "Corrected"],
                              horizontal=True, key="bg_signal", label_visibility="collapsed",
                              help="Corrected becomes available after a valid batch is applied in Background.")
            st.caption(f"{signal} · Applied: " + config_caption(background_state.applied_config))
        else:
            signal = "Input"
        background_state.signal = signal
        corrected = signal == "Corrected"
        weighted = bool(background_state.applied_results and
                        background_state.applied_config.intensity_mode == "energy_squared")
        controls = st.columns([1.3, 1, 1])
        display_modes = {"log10": "log10", "Linear": "linear", "Intensity × E²": "energy_squared"}
        mode_label = controls[0].selectbox(
            "Intensity display", list(display_modes), key="intensity_display",
            help="Intensity × E² multiplies unweighted spectra by energy loss squared (in meV²). Already weighted background results are not multiplied twice.")
        mode = display_modes[mode_label]
        x_min = controls[1].number_input("Energy min (meV)", value=-150.0, step=10.0, key="spectrum_x_min")
        x_max = controls[2].number_input("Energy max (meV)", value=150.0, step=10.0, key="spectrum_x_max")
        if x_min >= x_max:
            st.error("Energy min must be less than energy max.")
            render_config_download(config_slot)
            st.stop()
        x_limits = (x_min, x_max)
        with st.expander("Line appearance", expanded=False):
            st.caption("Choose a spectrum to customize. Styles also apply to SVG/PNG downloads and are saved in NumPy metadata.")
            # Keep styles separate from widget state so removing/reselecting a curve
            # or editing its label does not discard its appearance during this session.
            saved_styles = st.session_state.setdefault("curve_styles", {})
            curve_keys = [curve_identity_key(c) for c in curves]
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
        toggle_cols = st.columns(2)
        show_hover = toggle_cols[0].toggle("Show hover details", value=True, key="show_hover_details",
                               help="Show or hide the energy and intensity popup when hovering over the spectra.")
        show_detector = toggle_cols[1].toggle("Show detector preview", value=True, key="show_detector_preview",
                                  help="Show the click-to-position detector diffraction image beside the spectrum plot.")
        panel_settings = layout_settings()
        if ADJUSTABLE_PANELS_ENABLED:
            layout_help, layout_reset = st.columns([4, 1.4])
            layout_help.caption("Drag the divider to resize panel widths, the handles below the plots to change height, "
                                "or a panel title onto the other panel to swap positions." if show_detector else
                                "Drag the handle below the spectrum to change its height.")
            layout_reset.button("Reset panel layout", on_click=reset_panel_layout, width="stretch")
        st.session_state["spectrum_render_revision"] = st.session_state.get("spectrum_render_revision", 0) + 1
        if weighted or corrected:
            signal_array = "corrected" if corrected else "input"
            shown_curves = [dict(c, intensity=getattr(background_state.applied_results[curve_identity_key(c)], signal_array))
                            for c in curves]
        else:
            shown_curves = curves
        fig = go.Figure()
        for curve in shown_curves:
            style = curve["style"]
            fig.add_trace(go.Scatter(x=curve["energy"], y=spectrum_display_intensity(curve, mode, corrected, weighted),
                                     name=curve["label"], mode="lines", connectgaps=False,
                                     line=dict(color=style["color"], width=style["width"],
                                               dash=LINE_STYLES[style["line_style"]][0]),
                                     hovertemplate="%{y:.6g}<extra>%{fullData.name}</extra>"))
        fig.update_layout(height=panel_settings["panel_spectrum_height"], margin=dict(l=20, r=20, t=15, b=20), template="plotly_white", dragmode="pan",
                          paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified" if show_hover else False,
                          legend=dict(orientation="h", y=-0.2, x=0), uirevision="spectrum",
                          meta=dict(eels_render_revision=st.session_state["spectrum_render_revision"]),
                          # Preserve exploration across recalculation; explicit limit edits still apply.
                          xaxis=dict(title="Energy loss (meV)", range=x_limits, zerolinecolor="#bdcbd4",
                                     gridcolor="#EAF0F7", hoverformat=".4f", uirevision=json.dumps([x_limits, st.session_state.get("workspace_revision", 0)])),
                          yaxis=dict(title=intensity_label(mode, normalize, corrected, weighted), gridcolor="#EAF0F7",
                                     uirevision=f"yaxis:{mode}:{'weighted' if weighted else 'unweighted'}:{st.session_state.get('workspace_revision', 0)}"))
        workspace_row = st.container(key="workspace_row")
        with workspace_row:
            if show_detector:
                share = panel_settings["panel_detector_width"]
                if panel_settings["panel_detector_first"]:
                    detector_col, plot_col = st.columns([share, 1 - share], gap="medium")
                else:
                    plot_col, detector_col = st.columns([1 - share, share], gap="medium")
            else:
                detector_col, plot_col = None, st.container()
        with plot_col:
            with st.container(key="spectrum_panel"):
                st.markdown('<p class="eels-panel-title">EELS Spectrum</p>', unsafe_allow_html=True)
                with st.container(key="spectrum_axis_target"):
                    st.plotly_chart(fig, key="spectrum_plot", use_container_width=True, config={"displaylogo": False,
                                    "scrollZoom": True,
                                    "toImageButtonOptions": {"format": "svg", "filename": "eels_spectra"}})
                axis_scaling(key="spectrum_axis_scaling", height=0)
                st.caption("Click a legend entry to hide that curve · double-click a legend entry to "
                           "isolate it (hide all others) · double-click again to restore all.")
                if sigma_mev > 0:
                    st.caption(f"Gaussian broadening active: σ = {sigma_mev:g} meV (FWHM = {sigma_mev * np.sqrt(8 * np.log(2)):.3f} meV).")
                if mode == "energy_squared":
                    st.caption("Energy-squared display: I(E) × E², with E in meV. Data exports retain unscaled linear intensities.")
                if corrected and mode == "log10":
                    st.caption("Nonpositive corrected samples are masked in log10 display (gaps are not connected). Signed linear residuals remain in data exports.")
                if not any(np.any((c["energy"] >= x_min) & (c["energy"] <= x_max)) for c in curves):
                    st.warning("The selected energy window contains no data bins.")
        if detector_col is not None:
            with detector_col:
                with st.container(key="detector_panel"):
                    st.markdown('<p class="eels-panel-title">Detector</p>', unsafe_allow_html=True)
                    curve = curves[preview_index]
                    info = scans[curve["path"]]
                    axis = curve["energy"]
                    _, raw_index = nearest_energy_index(axis, requested_energy,
                                                        unshifted=ordering == "Unshifted FFT")
                    try:
                        pattern = cached_diffraction_pattern(info, raw_index, curve["dummy"], curve["probe_x"], curve["probe_y"])
                        if not np.isfinite(pattern).all():
                            raise ValueError("Diffraction plane contains NaN or infinite intensities")
                        if pattern_log:
                            # Avoids materializing a copy of every positive value just for its minimum.
                            positive_min = np.min(pattern, where=pattern > 0, initial=np.inf)
                            if not np.isfinite(positive_min):
                                raise ValueError("This plane has no positive intensities. Turn off log diffraction image.")
                            shown = np.log10(np.clip(pattern, positive_min, None))
                        else:
                            shown = pattern
                        center = detector_center(info, offset_px, offset_py)
                        st.session_state["detector_render_revision"] = st.session_state.get("detector_render_revision", 0) + 1
                        image = go.Figure(go.Heatmap(z=shown, colorscale="Viridis", showscale=False,
                                                     hovertemplate="py=%{x}<br>px=%{y}<br>%{z:.5g}<extra></extra>"))
                        image.add_shape(type="circle", x0=center[1] - radius, x1=center[1] + radius,
                                        y0=center[0] - radius, y1=center[0] + radius,
                                        line=dict(color="#ff765e", width=2))
                        # Draw the center as shapes so it cannot steal clicks from nearby pixels.
                        image.add_shape(type="line", x0=center[1] - 2, x1=center[1] + 2,
                                        y0=center[0], y1=center[0], line=dict(color="#ff765e", width=2))
                        image.add_shape(type="line", x0=center[1], x1=center[1],
                                        y0=center[0] - 2, y1=center[0] + 2, line=dict(color="#ff765e", width=2))
                        # Keep pixels square by shrinking the plotting domain, not by
                        # repeatedly expanding coordinate ranges when the panel resizes.
                        margin = dict(l=10, r=10, t=10, b=10)
                        # A shared revision keyed on the plane's shape preserves exploration across
                        # recalculation but resets it if a differently-sized detector plane loads.
                        # New revision discards zoom ranges inflated by the old resize behavior.
                        axis_revision = f"detector-domain:{pattern.shape}:{st.session_state.get('workspace_revision', 0)}"
                        image.update_layout(autosize=True,
                                            height=panel_settings["panel_detector_height"],
                                            template="plotly_white", margin=margin, paper_bgcolor="white",
                                            plot_bgcolor="white", uirevision="detector", dragmode="pan",
                                            meta=dict(eels_render_revision=st.session_state["detector_render_revision"]),
                                            xaxis=dict(range=[-0.5, pattern.shape[1] - 0.5], uirevision=axis_revision,
                                                      visible=False, constrain="domain"),
                                            yaxis=dict(range=[-0.5, pattern.shape[0] - 0.5], visible=False,
                                                      constrain="domain", scaleanchor="x", scaleratio=1,
                                                      uirevision=axis_revision))
                        with st.container(key="detector_click_target"):
                            st.plotly_chart(image, key="detector_preview", use_container_width=True,
                                            config={"displaylogo": False, "scrollZoom": True,
                                                    "toImageButtonOptions": {"format": "png", "filename": "eels_detector"}})
                        axis_scaling(key="detector_axis_scaling", height=0, data=dict(
                            selector=".st-key-detector_click_target .js-plotly-plot",
                            viewport_key="eels.detector.viewport"))
                        preview_id = json.dumps([info.path, info.mtime_ns, info.revision, curve["dummy"], curve["probe_x"], curve["probe_y"], raw_index])
                        st.session_state["detector_preview_context"] = dict(preview_id=preview_id, shape=pattern.shape,
                                                                           selected_shapes=[i.shape[-2:] for i in infos])
                        # Renew the bridge after each Streamlit redraw, including manual edits.
                        st.session_state["detector_click_revision"] = st.session_state.get("detector_click_revision", 0) + 1
                        detector_click_bridge(key="detector_click", data={"preview_id": preview_id,
                                              "revision": st.session_state["detector_click_revision"]},
                                              on_clicked_change=move_detector_from_click, height=0)
                        if any(c - radius < 0 or c + radius > n - 1 for c, n in zip(center, pattern.shape)):
                            st.warning("The detector extends beyond this array. Only pixels inside the recorded plane are integrated.")
                    except (OSError, ValueError, IndexError, EOFError) as exc:
                        st.warning(f"Preview unavailable: {exc}")
                    # Editable detector center/radius, always available even if the preview above failed.
                    edit_cols = st.columns(3)
                    edit_cols[0].number_input("px", min_value=-(min_px // 2), max_value=(min_px - 1) // 2,
                                              value=offset_px, step=1, key="offset_px")
                    edit_cols[1].number_input("py", min_value=-(min_py // 2), max_value=(min_py - 1) // 2,
                                              value=offset_py, step=1, key="offset_py")
                    edit_cols[2].number_input("radius", min_value=0.1, value=radius, step=1.0, key="detector_radius")
                    st.button("Reset detector center", on_click=reset_detector_center, use_container_width=True)
        if panel_layout is not None:
            panel_layout(key="panel_layout_bridge", height=0,
                         data=dict(panel_settings, render_revision=st.session_state["spectrum_render_revision"]),
                         on_layout_change=update_panel_layout)
        settings["plot"] = dict(mode=mode, x_limits=x_limits, show_hover_details=show_hover)
        with st.expander("Export spectra & figures", expanded=True):
            data_stage = "energy²-weighted linear intensities" if weighted else "linear intensities"
            st.caption(f"Data exports contain the full energy range and {data_stage}. Figures use the display controls above; browser-only zoom and legend changes are not applied.")
            if sigma_mev > 0:
                st.caption(f"All downloads include Gaussian broadening (σ = {sigma_mev:g} meV). Turn broadening off to export unbroadened spectra; NPZ records the processing settings.")
            png_dpi = st.selectbox("PNG export resolution", PNG_DPI_OPTIONS, index=PNG_DPI_OPTIONS.index(300),
                                   format_func=lambda d: f"{d} DPI", key="png_dpi_spectrum",
                                   help="Resolution used for the PNG figure download below.")
            if background_state.applied_results:
                st.caption(f"Exported signal: {signal}. Unfitted bins are missing (NaN), never zero-filled.")
            if background_state.applied_results:
                csv_data = background_exports.background_csv(curves, settings, background_state, signal, all_arrays=False)
                npz_data = background_exports.background_npz(curves, settings, background_state, signal)
                export_stem = f"eels_spectra_{signal.lower()}"
            else:
                csv_data, npz_data = cached_export_csv(curves), cached_export_npz(curves, settings)
                export_stem = "eels_spectra"
            exports = st.columns(5)
            exports[0].download_button("CSV data", csv_data, export_stem + ".csv", "text/csv", use_container_width=True)
            exports[1].download_button("NumPy + settings", npz_data, export_stem + ".npz",
                                       "application/octet-stream", use_container_width=True)
            partial_corrected = corrected and any(not r.validity_mask.all() for r in background_state.applied_results.values())
            if partial_corrected:
                exports[2].caption("Partial corrected fits require masked NPZ. Select Input for the original notebook .npy export.")
            elif len(bins) == 1:
                exports[2].download_button("Notebook .npy", cached_notebook_npy(shown_curves), "eels_spectra_corrected.npy" if corrected else "eels_spectra.npy", "application/octet-stream", use_container_width=True)
            else:
                exports[2].caption("Use NPZ for unequal energy lengths.")
            figures = figure_downloads(shown_curves, mode, x_limits, normalize, png_dpi, corrected, weighted)
            exports[3].download_button("SVG figure", figures["svg"], export_stem + ".svg", "image/svg+xml", use_container_width=True)
            exports[4].download_button("PNG figure", figures["png"], export_stem + ".png", "image/png", use_container_width=True)

            if background_state.applied_results:
                background_downloads = st.columns(2)
                background_downloads[0].download_button("Background results CSV", background_exports.background_csv(curves, settings, background_state, signal),
                    "eels_background.csv", "text/csv", use_container_width=True)
                background_downloads[1].download_button("Background results NPZ", npz_data,
                    "eels_background.npz", "application/octet-stream", use_container_width=True)

    with background_tab:
        render_background(curves, background_state)

    with angle_tab:
        render_angle_resolved(curves, scans, settings)

    with details_tab:
        st.dataframe([{"Label": label, "File": Path(info.path).name, "Shape": str(info.shape),
                       "Type": info.dtype, "Size (MB)": round(info.size / 1e6, 1)}
                      for info, label in zip(infos, labels)], use_container_width=True, hide_index=True)
        st.markdown("""
**Array conventions**

- 3D: `(energy, px, py)`, one probe block per file.
- 6D: `(dummy, energy, probe_x, probe_y, px, py)`, first dummy index and selected probe positions.
- The circular detector uses `(px − center_px)² + (py − center_py)² ≤ radius²`.
- Full-probe normalization divides by the sum over **all energy bins and all pixels** at that probe.
- Energy is `fftshift(fftfreq(n_energy, timestep_fs × stride / 1000)) × 4.13566769692386` in meV.
- Input intensities are assumed FFT-shifted by default, as in the notebook. Choose unshifted only if your simulation output uses raw FFT ordering.
- Optional Gaussian broadening operates on linear EELS intensity after detector integration and before log display. σ is entered in meV and converted to bins using each file's energy spacing. The kernel extends to 4σ; reflecting boundaries preserve the recorded sum without wrapping between energy endpoints. Edge features can be affected by this boundary assumption.

File headers are inspected without memory mapping. Selected data is read directly in small blocks, avoiding a full-file virtual-memory reservation. Changing the plot or preview reuses cached spectra.
Different energy lengths are supported; shared time step and stride must be appropriate for every selected scan.
""")
        st.code(json.dumps(settings, indent=2), language="json")
