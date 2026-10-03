"""Background tab for the existing EELS Studio interface."""
import numpy as np
from eels_studio.ui.plotting import go, LINE_STYLES
import streamlit as st

from eels_studio.core.background import ANALYTIC_MODELS, BACKGROUND_MODELS, BackgroundConfig, auto_segments_from_peaks, background_input, default_analytic_segments, initial_bounds
from eels_studio.ui.plotting import config_caption
from eels_studio.ui.cache_layer import cached_background_fit
from eels_studio.core.identity import curve_identity_key
from eels_studio.ui.components.axis_scaling import register_axis_scaling
from eels_studio.ui.components.background_click import render_boundary_picker, reset_boundary_pick
from eels_studio.core.integration import integrate_area, symmetric_bounds, normalize_to_original


def _select_pick_target(target):
    reset_boundary_pick()
    st.session_state["bg_pick_target"] = target


def _fit_pick_button(group, label, *, disabled=False):
    active = not disabled and st.session_state.get("bg_pick_target") == group
    st.button(f"Cancel {label}" if active else f"Pick {label}", key=f"bg_pick_{group}",
              use_container_width=True, disabled=disabled,
              on_click=_select_pick_target, args=("" if active else group,),
              help="Click the minimum, then the maximum in the plot. Dragging still pans.")
    return st.empty()


PEAK_COLORS = ("#d88716", "#8157b6", "#d05f60", "#237ca9", "#6b8f32")


def _enable_area(peak_id):
    st.session_state[f"bg_area_selected_{peak_id}"] = True


def _add_peak():
    peak_id = st.session_state["bg_next_peak_id"]
    st.session_state["bg_next_peak_id"] = peak_id + 1
    st.session_state["bg_peak_ids"].append(peak_id)
    st.session_state[f"bg_area_center_{peak_id}"] = st.session_state["bg_peak_default_center"]
    st.session_state[f"bg_area_half_width_{peak_id}"] = st.session_state["bg_peak_default_half_width"]
    st.session_state[f"bg_area_selected_{peak_id}"] = False


def _remove_peak(peak_id):
    st.session_state["bg_peak_ids"].remove(peak_id)
    if st.session_state.get("bg_pick_target") == f"area:{peak_id}":
        st.session_state["bg_pick_target"] = ""
        reset_boundary_pick()


def _sync_lambda(source, target):
    st.session_state[target] = st.session_state[source]


def _apply_auto_segments(curve, n_segments, low, high, mode):
    # Runs before the script reruns, so the segment number_inputs (instantiated
    # below with the same keys) pick up these values instead of their defaults.
    try:
        detected = auto_segments_from_peaks(curve["energy"],
            background_input(curve["energy"], curve["intensity"], mode), n_segments, low, high)
    except ValueError as exc:
        st.session_state["bg_auto_segments_error"] = str(exc)
        return
    st.session_state.pop("bg_auto_segments_error", None)
    for i, (lo, hi) in enumerate(detected):
        st.session_state[f"bg_seg_lo_{i}"] = lo
        st.session_state[f"bg_seg_hi_{i}"] = hi


def _intensity_view_range(arrays, *, include_zero=False):
    """Padded limits from the visible samples, without altering their values."""
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if include_zero:
        values = np.append(values, 0.0)
    if not len(values):
        return None
    low, high = float(values.min()), float(values.max())
    padding = (high - low) * 0.06 if high > low else (abs(low) * 0.06 or 1.0)
    return [low - padding, high + padding]


def preview_figure(curve, result=None, mode="linear", *, view_bounds=None, view_revision=0):
    fig = go.Figure()
    input_values = background_input(curve["energy"], curve["intensity"], mode)
    fig.add_trace(go.Scatter(x=curve["energy"], y=input_values, name="Input (after broadening)",
                            mode="lines", connectgaps=False, line=dict(color="#88949e", width=1.5)))
    if result is not None:
        fit_segments = result.diagnostics.fit_segments
        if fit_segments:
            for lo, hi in fit_segments:
                fig.add_vrect(x0=lo, x1=hi, fillcolor="#137c78", opacity=0.10,
                              line_width=0)
        else:
            low, high = result.diagnostics.actual_bounds
            fig.add_vrect(x0=low, x1=high, fillcolor="#137c78", opacity=0.08,
                          line_width=0)
        if result.diagnostics.valid:
            fig.add_trace(go.Scatter(x=result.energy, y=result.baseline, name="Estimated baseline (dashed)",
                mode="lines", connectgaps=False, line=dict(color="#485868", width=2, dash="dash")))
            style = curve["style"]
            dash = LINE_STYLES[style["line_style"]][0]
            fig.add_trace(go.Scatter(x=result.energy, y=result.corrected, name="Corrected (input − baseline)",
                mode="lines", connectgaps=False, line=dict(color=style["color"], width=style["width"], dash=dash)))
    fig.add_hline(y=0, line_color="#88949e", line_width=1)
    identity = "background:" + curve_identity_key(curve)
    fig.update_layout(height=590, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=15, b=15), hovermode="x unified", dragmode="pan",
                      legend=dict(orientation="h", y=1.13, x=0),
                      uirevision=identity)
    fig.update_yaxes(title_text="Intensity × E² (meV²)" if mode == "energy_squared" else "Intensity")
    # Explicit view actions reset saved ranges; fitting and ordinary reruns do not.
    view_identity = f"{identity}:view:{view_revision}"
    fig.update_yaxes(uirevision=f"{view_identity}:{mode}", zeroline=True)
    fig.update_xaxes(uirevision=view_identity)
    if view_bounds is not None:
        low, high = view_bounds
        visible = (curve["energy"] >= low) & (curve["energy"] <= high)
        intensities = [input_values[visible]]
        if result is not None and result.diagnostics.valid:
            intensities.extend([result.baseline[visible], result.corrected[visible]])
        intensity_range = _intensity_view_range(intensities, include_zero=True)
        if intensity_range is not None:
            fig.update_yaxes(range=intensity_range, autorange=False)
        fig.update_xaxes(range=[low, high], autorange=False)
    fig.update_xaxes(title_text="Energy loss (meV)")
    return fig


def render_background(curves, state):
    st.caption("Fit the selected intensity form after optional Gaussian broadening. "
               "Changing Gaussian σ or intensity form requires a new fit. Plot zoom never changes the fitting domain.")
    st.markdown("""<style>
    @media (max-width: 900px) {
      .st-key-background_layout [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
      .st-key-background_layout [data-testid="stColumn"] {min-width: 100%;}
      .st-key-background_layout [class*="st-key-background_peak_"] [data-testid="stHorizontalBlock"] {flex-wrap: nowrap;}
      .st-key-background_layout [class*="st-key-background_peak_"] [data-testid="stColumn"] {min-width: 0;}
    }
    </style>""", unsafe_allow_html=True)
    with st.container(key="background_layout"):
        left, right = st.columns([3, 7], gap="medium")
        identities = [curve_identity_key(c) for c in curves]
        labels = dict(zip(identities, [c["label"] for c in curves]))
        with right:
            if st.session_state.get("bg_preview_curve") not in identities:
                st.session_state["bg_preview_curve"] = identities[0]
            selector_col, display_col = st.columns([2, 1], gap="small")
            selected = selector_col.selectbox("Preview-spectrum selector", identities,
                format_func=labels.get, key="bg_preview_curve")
            display = display_col.selectbox("Background intensity", ["Linear", "Intensity × E²"], key="bg_display",
                help="Choose the signal used for preview, fitting, subtraction, peak areas, and background exports. E is energy loss in meV.")
            mode = "energy_squared" if display == "Intensity × E²" else "linear"
        curve = curves[identities.index(selected)]
        with left, st.container(key="background_method_panel"):
            method_panel = st.expander("Background method", expanded=False)
        with left, st.container(key="background_area_panel"):
            area_panel = st.expander("Peak area", expanded=True)
        with method_panel:
            method = st.selectbox("Method", ["arPLS", "SNIP", *ANALYTIC_MODELS], key="bg_method")
            low, high = initial_bounds(curves)
            fit_pick_notes = {}
            if method in ANALYTIC_MODELS:
                spec = BACKGROUND_MODELS[method]
                n_segments = st.number_input("Fit segments", min_value=2, max_value=4, value=2, step=1,
                    key="bg_n_segments",
                    help="Flanking energy windows that exclude the peak; the background is fit only on "
                         "these, then evaluated and subtracted across their full span (first segment's "
                         "start to last segment's end), including the peak region between them.")
                st.button("Auto-detect segments from peak", key="bg_auto_segments", use_container_width=True,
                    help="Finds the most prominent local peak(s) in the preview spectrum (at least a ~15% "
                         "local rise) and places segments to flank them. A starting point, not a substitute "
                         "for checking the preview plot — it can miss a peak below that threshold or pick "
                         "the wrong one for a noisy or multi-featured spectrum.",
                    on_click=_apply_auto_segments, args=(curve, n_segments, low, high, mode))
                if "bg_auto_segments_error" in st.session_state:
                    st.warning(st.session_state.pop("bg_auto_segments_error"))
                defaults = default_analytic_segments(low, high, n_segments)
                segments = []
                for i in range(n_segments):
                    # Seed via session_state (not value=) since Auto-detect also writes these
                    # keys; passing both triggers a Streamlit widget-policy warning.
                    st.session_state.setdefault(f"bg_seg_lo_{i}", defaults[i][0])
                    st.session_state.setdefault(f"bg_seg_hi_{i}", defaults[i][1])
                    fit_pick_notes[f"segment_{i}"] = _fit_pick_button(f"segment_{i}", f"Segment {i + 1}")
                    cols = st.columns(2)
                    seg_lo = cols[0].number_input(f"Segment {i + 1} min (meV)", key=f"bg_seg_lo_{i}", format="%.6g")
                    seg_hi = cols[1].number_input(f"Segment {i + 1} max (meV)", key=f"bg_seg_hi_{i}", format="%.6g")
                    segments.append((seg_lo, seg_hi))
                minimum, maximum = min(lo for lo, _ in segments), max(hi for _, hi in segments)
                energy_factor = st.number_input("Energy scale factor (meV)", min_value=1e-6, value=40.0,
                    key="bg_energy_factor",
                    help="Energies are divided by this before fitting (numerical conditioning), "
                         "matching the source toolkit.")
                with st.expander("Start point & bounds", expanded=False):
                    st.caption("Defaults for power/power0 mirror the source toolkit's STO example; "
                               "exppoly/pVoigt defaults are generic starting points that need tuning "
                               "per dataset.")
                    model_start, model_lower, model_upper = [], [], []
                    for i, name in enumerate(spec.params):
                        c1, c2, c3 = st.columns(3)
                        model_start.append(c1.number_input(f"{name} start", value=spec.start[i],
                            key=f"bg_mstart_{method}_{name}", format="%.6g"))
                        model_lower.append(c2.number_input(f"{name} lower bound", value=spec.bounds[0][i],
                            key=f"bg_mlower_{method}_{name}", format="%.6g"))
                        model_upper.append(c3.number_input(f"{name} upper bound", value=spec.bounds[1][i],
                            key=f"bg_mupper_{method}_{name}", format="%.6g"))
                config = BackgroundConfig(method=method, domain="selected", energy_min=minimum, energy_max=maximum,
                    segments=tuple(segments), model_start=tuple(model_start), model_lower=tuple(model_lower),
                    model_upper=tuple(model_upper), energy_factor=energy_factor, intensity_mode=mode)
            else:
                domain = st.selectbox("Fit domain", ["Selected energy interval", "Full recorded spectrum"], key="bg_domain")
                fit_pick_notes["fit"] = _fit_pick_button("fit", "Fit interval",
                    disabled=domain == "Full recorded spectrum")
                minimum = st.number_input("Fit energy minimum (meV)", value=low, format="%.6f", key="bg_min",
                                          disabled=domain == "Full recorded spectrum")
                maximum = st.number_input("Fit energy maximum (meV)", value=high, format="%.6f", key="bg_max",
                                          disabled=domain == "Full recorded spectrum")
                st.caption("Initial bounds use common positive-energy coverage where possible. The first positive bin "
                           "is not automatically beyond the zero-loss tail. At least eight fitted bins are required.")
                log_lambda, tolerance, iterations, window = 5.0, 1e-3, 100, 10.0
                if method == "arPLS":
                    st.session_state.setdefault("bg_lambda_slider", 5.0)
                    st.session_state.setdefault("bg_lambda_number", 5.0)
                    st.slider("log10(λ)", 2.0, 10.0, step=0.1, key="bg_lambda_slider",
                              on_change=_sync_lambda, args=("bg_lambda_slider", "bg_lambda_number"))
                    log_lambda = st.number_input("log10(λ), numeric", min_value=2.0, max_value=10.0, step=0.1,
                        key="bg_lambda_number", on_change=_sync_lambda, args=("bg_lambda_number", "bg_lambda_slider"))
                    st.caption(f"λ = {10 ** log_lambda:g}. Larger λ gives a smoother baseline. λ = 1e5 is a starting point, not a TACAW optimum.")
                else:
                    window = st.number_input("Maximum half-window (meV)", min_value=0.000001, value=10.0,
                                             key="bg_window", format="%.6f")
                    st.caption("Rounded to the nearest bin separately on each grid (half ties up). Linear intensity; no log transform.")
                with st.expander("Advanced settings", expanded=False):
                    if method == "arPLS":
                        tolerance = st.number_input("Tolerance", min_value=1e-12, max_value=0.999,
                                                    value=1e-3, format="%.6g", key="bg_tolerance")
                        iterations = st.number_input("Maximum iterations", min_value=1, value=100, step=1, key="bg_iterations")
                        st.caption("Second-order differences; convergence is required for application.")
                    else:
                        st.caption("decreasing=True · filter_order=2 · no extra smoothing")
                config = BackgroundConfig(method=method, domain="selected" if domain.startswith("Selected") else "full",
                    energy_min=minimum if domain.startswith("Selected") else None,
                    energy_max=maximum if domain.startswith("Selected") else None,
                    log10_lambda=log_lambda, tolerance=tolerance, max_iterations=iterations, half_window_mev=window,
                    intensity_mode=mode)
            state.draft = config
            preview_key = (selected, state.fingerprints[selected], config)
            if state.preview_key != preview_key:
                state.preview, state.preview_key = None, None
            if state.applied_config is not None and state.applied_config != config:
                st.caption("Unapplied edits — main spectra and exports still use the applied settings.")
            if st.button("Preview", key="bg_preview", use_container_width=True):
                # A failed explicit refresh must never leave an older preview visible.
                state.preview, state.preview_key = None, None
                try:
                    state.preview = cached_background_fit(curve["energy"], curve["intensity"], config)
                    state.preview_key = preview_key
                except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    st.error(str(exc))
            if st.button("Apply to all spectra", key="bg_apply", use_container_width=True):
                with st.spinner("Fitting each selected spectrum independently…"):
                    failures = state.apply(curves, config, cached_background_fit)
                if failures:
                    st.error("Batch not applied. Previous applied results, if any, are retained.")
                    for identity, error in failures.items():
                        st.error(f"{labels[identity]}: {error}")
                else:
                    st.session_state["bg_next_signal"] = "Corrected"
                    st.rerun()
            if st.button("Reset background", key="bg_reset", use_container_width=True):
                state.reset()
                st.session_state["bg_next_signal"] = "Input"
                st.rerun()
        with right:
            result = state.preview
            plot_slot = st.empty()
            if result is None:
                if method in ANALYTIC_MODELS:
                    requested = f"{len(segments)} segments spanning {minimum:.6g}–{maximum:.6g} meV"
                else:
                    requested = (f"{minimum:.6g}–{maximum:.6g} meV" if config.domain == "selected"
                                 else "full recorded spectrum")
                st.caption(f"Draft fit interval: {requested} · No matching preview. Press Preview to fit.")
            else:
                d = result.diagnostics
                if d.fit_segments:
                    seg_text = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in d.fit_segments)
                    st.caption(f"Segments [{seg_text}] meV · {d.fitted_bins} points fit · "
                               f"span {d.actual_bounds[0]:.6g}–{d.actual_bounds[1]:.6g} meV · solver: {d.status}")
                else:
                    st.caption(f"Requested: {d.requested_bounds[0]:.6g}–{d.requested_bounds[1]:.6g} meV · "
                               f"actual bin-aligned: {d.actual_bounds[0]:.6g}–{d.actual_bounds[1]:.6g} meV · "
                               f"{d.fitted_bins} fitted bins · solver: {d.status}")
            recorded = (float(curve["energy"][0]), float(curve["energy"][-1]))
            focus = (minimum, maximum) if config.domain == "selected" else recorded
            valid_focus = (np.isfinite(focus).all() and recorded[0] <= focus[0] < focus[1] <= recorded[1]
                           and np.any((curve["energy"] >= focus[0]) & (curve["energy"] <= focus[1])))
            focus_column, full_column = st.columns(2)
            with focus_column:
                focus_clicked = st.button("Focus fit interval", key="bg_focus_view", disabled=not valid_focus,
                    use_container_width=True, help="View only: zoom the energy axis and rescale intensities within the fit interval.")
            with full_column:
                full_clicked = st.button("Show full spectrum", key="bg_full_view", use_container_width=True,
                    help="View only: show the recorded energy domain and rescale intensities.")
            if focus_clicked or full_clicked:
                st.session_state["bg_preview_view"] = "fit" if focus_clicked else "full"
                st.session_state["bg_view_revision"] = st.session_state.get("bg_view_revision", 0) + 1
            view_bounds = (focus if valid_focus and st.session_state.get("bg_preview_view", "fit") == "fit"
                           else recorded)
            area_context = (selected, recorded)
            samples = curve["energy"]
            if valid_focus:
                samples = samples[(samples >= focus[0]) & (samples <= focus[1])]
            default_center = float((samples[0] + samples[-1]) / 2)
            default_half_width = max(float((samples[-1] - samples[0]) / 4), 1e-6)
            st.session_state["bg_peak_default_center"] = default_center
            st.session_state["bg_peak_default_half_width"] = default_half_width
            peak_ids = st.session_state.setdefault("bg_peak_ids", [1, 2])
            st.session_state.setdefault("bg_next_peak_id", max(peak_ids, default=0) + 1)
            new_spectrum = st.session_state.get("bg_area_context") != area_context
            for peak_id in peak_ids:
                center_key = f"bg_area_center_{peak_id}"
                width_key = f"bg_area_half_width_{peak_id}"
                selected_key = f"bg_area_selected_{peak_id}"
                if new_spectrum:
                    st.session_state[center_key] = default_center
                    st.session_state[width_key] = default_half_width
                    st.session_state[selected_key] = False
                else:
                    # Keep an existing single-peak selection when this UI first appears.
                    st.session_state.setdefault(center_key, st.session_state.get("bg_area_center", default_center)
                                                if peak_id == 1 else default_center)
                    st.session_state.setdefault(width_key, st.session_state.get("bg_area_half_width", default_half_width)
                                                if peak_id == 1 else default_half_width)
                    st.session_state.setdefault(selected_key, st.session_state.get("bg_area_selected", False)
                                                if peak_id == 1 else False)
            st.session_state["bg_area_context"] = area_context
            pick_targets, pick_labels = {}, {}
            if method in ANALYTIC_MODELS:
                for i in range(n_segments):
                    group = f"segment_{i}"
                    pick_targets[group] = (f"bg_seg_lo_{i}", f"bg_seg_hi_{i}")
                    pick_labels[group] = f"Segment {i + 1}"
            elif config.domain == "selected":
                pick_targets = {"fit": ("bg_min", "bg_max")}
                pick_labels = {"fit": "Fit interval"}
            for number, peak_id in enumerate(peak_ids, 1):
                group = f"area:{peak_id}"
                pick_targets[group] = (f"bg_area_center_{peak_id}",)
                pick_labels[group] = f"Peak {number} center"
            if st.session_state.get("bg_pick_target") not in pick_targets:
                st.session_state["bg_pick_target"] = ""
            pick_context = repr((selected, state.fingerprints[selected], method, config.domain, tuple(pick_targets.items())))
            if st.session_state.get("bg_pick_context") != pick_context:
                reset_boundary_pick()
            st.session_state["bg_pick_context"] = pick_context
            st.session_state["bg_pick_targets"] = pick_targets
            st.session_state["bg_pick_bounds"] = recorded
            pick_target = st.session_state.get("bg_pick_target", "")
            boundary_target = ""
            if pick_target:
                step = st.session_state.get("bg_pick_step", 0)
                boundary_target = pick_targets[pick_target][step]
                if not pick_target.startswith("area:"):
                    instruction = "First click: set the minimum." if step == 0 else "Second click: set the maximum."
                    fit_pick_notes[pick_target].caption(
                        f"{pick_labels[pick_target]} — {instruction} Click anywhere inside the plot.")
            completed = st.session_state.get("bg_pick_completed")
            if completed and completed[0] in pick_labels and not completed[0].startswith("area:"):
                fit_pick_notes[completed[0]].caption(
                    f"{pick_labels[completed[0]]} set: {completed[1]:.6g}–{completed[2]:.6g} meV. "
                    "Choose another segment or press Preview to fit.")
            if "bg_pick_error" in st.session_state:
                with area_panel if pick_target.startswith("area:") else method_panel:
                    st.warning(st.session_state.pop("bg_pick_error"))
            figure = preview_figure(curve, result, mode, view_bounds=view_bounds,
                                    view_revision=st.session_state.get("bg_view_revision", 0))
            for number, peak_id in enumerate(peak_ids, 1):
                group = f"area:{peak_id}"
                color = PEAK_COLORS[(number - 1) % len(PEAK_COLORS)]
                with area_panel, st.container(key=f"background_peak_{peak_id}", border=True):
                    title, pick, remove = st.columns([1, 1.5, 1], gap="small")
                    title.markdown(f"<span style='color:{color}'>●</span> **Peak {number}**",
                                   unsafe_allow_html=True)
                    pick.button("Cancel pick" if pick_target == group else "Pick center",
                                key=f"bg_pick_area_{peak_id}", use_container_width=True,
                                on_click=_select_pick_target, args=("" if pick_target == group else group,),
                                help="Click once in the plot to set this peak's center.")
                    remove.button("Remove", key=f"bg_remove_peak_{peak_id}", use_container_width=True,
                                  on_click=_remove_peak, args=(peak_id,))
                    if pick_target == group:
                        st.caption(f"Peak {number} center — click once in the plot.")
                    center_col, width_col = st.columns(2, gap="small")
                    center = center_col.number_input("Center (meV)", key=f"bg_area_center_{peak_id}",
                                                     format="%.6g", on_change=_enable_area, args=(peak_id,))
                    half_width = width_col.number_input("Half-width (± meV)",
                        key=f"bg_area_half_width_{peak_id}", min_value=0.000001, step=1.0, format="%.6g",
                        on_change=_enable_area, args=(peak_id,),
                        help="Distance on each side of this peak's center.")
                    area_min, area_max = symmetric_bounds(center, half_width)
                    selected_area = st.session_state.get(f"bg_area_selected_{peak_id}", False)
                    if selected_area:
                        figure.add_vrect(x0=area_min, x1=area_max, fillcolor=color, opacity=0.13,
                                         line_width=1, line_color=color, layer="below",
                                         name=f"Peak {number} area")
                        figure.add_vline(x=center, line_color=color, line_dash="dot", line_width=1,
                                         name=f"Peak {number} center")
                    if selected_area and (result is None or not result.diagnostics.valid):
                        st.info("Press Preview to fit the background before calculating corrected area.")
                    elif selected_area:
                        try:
                            area = integrate_area(result.energy, result.corrected, area_min, area_max)
                        except ValueError as exc:
                            st.warning(str(exc))
                        else:
                            area_units = "intensity × meV³" if mode == "energy_squared" else "intensity × meV"
                            st.write(f"Area: {area:.6g} {area_units}")
                            try:
                                _, fraction = normalize_to_original(area, result.energy, result.input)
                            except ValueError as exc:
                                st.write("Normalized area: Unavailable")
                                st.warning(str(exc))
                            else:
                                st.write("Normalized area: Unavailable" if fraction is None
                                         else f"Normalized area: {fraction:.6g}")
            with area_panel:
                st.button("Add peak", key="bg_add_peak", on_click=_add_peak, use_container_width=True)
            st.caption("The initial view focuses on the fit interval and scales intensity within it. "
                       "Use Focus fit interval to restore that view after zooming. View controls never refit the baseline.")
            revision = st.session_state.get("background_render_revision", 0) + 1
            st.session_state["background_render_revision"] = revision
            # Refresh numerical traces independently of uirevision, which preserves zoom.
            figure.update_layout(datarevision=revision, meta=dict(eels_render_revision=revision))
            plot_slot.plotly_chart(figure, key="background_preview_plot",
                            use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
            render_boundary_picker(context=pick_context, target=boundary_target)
            axis_scaling = register_axis_scaling()
            axis_scaling(key="background_axis_scaling", height=0, data=dict(
                selector=".st-key-background_preview_plot .js-plotly-plot",
                viewport_key="eels.background.viewport"))
            if result is None:
                st.caption("Press Preview to fit this spectrum with the draft settings. The preview is independent of applied results.")
            else:
                d = result.diagnostics
                if not d.valid:
                    st.warning("This preview is invalid and cannot be applied.")
                if d.half_window_bins is not None:
                    st.caption(f"SNIP half-window: {d.half_window_bins} bins = {d.effective_half_window_mev:.6g} meV. No convergence statistic is used.")
                elif d.tol_history:
                    st.caption(f"Final tolerance: {d.tol_history[-1]:.4g} · {len(d.tol_history)} iterations")
            bounds = [(c["energy"][0], c["energy"][-1]) for c in curves] if config.domain == "full" else [(minimum, maximum)]
            if any(a <= 0 <= b for a, b in bounds):
                st.warning("The fitting domain contains zero energy: no zero-loss exclusion has been performed.")
            if config.domain == "full":
                st.caption("Each spectrum uses its own full recorded domain; coverage may differ.")
                st.dataframe([dict(Spectrum=c["label"], Minimum_meV=float(c["energy"][0]),
                                   Maximum_meV=float(c["energy"][-1])) for c in curves], hide_index=True)
            spacings = [c["energy"][1] - c["energy"][0] for c in curves if len(c["energy"]) > 1]
            if spacings and not np.allclose(spacings, spacings[0], rtol=1e-8, atol=0):
                st.warning("Energy spacings differ. The same numerical arPLS λ does not imply identical smoothing in energy units. Original grids are preserved without interpolation.")
            if state.applied_results:
                st.caption("Applied: " + config_caption(state.applied_config))
                st.dataframe([dict(Spectrum=c["label"], Requested_meV=str(state.applied_results[identity].diagnostics.requested_bounds),
                    Actual_meV=str(state.applied_results[identity].diagnostics.actual_bounds),
                    Bins=state.applied_results[identity].diagnostics.fitted_bins,
                    Status=state.applied_results[identity].diagnostics.status,
                    SNIP_bins=state.applied_results[identity].diagnostics.half_window_bins,
                    SNIP_meV=state.applied_results[identity].diagnostics.effective_half_window_mev)
                    for c, identity in zip(curves, identities)], hide_index=True)
                if state.applied_config.method in ANALYTIC_MODELS:
                    param_names = state.applied_results[identities[0]].diagnostics.model_param_names
                    st.caption("Fitted model coefficients (± approx. 95% CI, normal approximation — not "
                               "bit-identical to MATLAB's t-distribution-based nonlinear-regression CI):")
                    st.dataframe([dict(Spectrum=c["label"], **{
                            name: f"{state.applied_results[identity].diagnostics.model_coeffs[i]:.4g} ± "
                                  f"{state.applied_results[identity].diagnostics.model_coeff_errors[i]:.2g}"
                            for i, name in enumerate(param_names)})
                        for c, identity in zip(curves, identities)], hide_index=True)
    st.caption("An empirical baseline may remove genuine broad vibrational intensity. Subtraction is an analysis choice, "
               "not automatic identification of nonphysical background. Check parameter and interval sensitivity; "
               "this does not correct Fourier leakage or validate multiphonon intensity.")
