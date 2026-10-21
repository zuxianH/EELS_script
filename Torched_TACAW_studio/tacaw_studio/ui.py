"""Streamlit UI. Scientific runtime dependencies are loaded by subprocesses."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd  # Initialize before Plotly validation, as in EELS Studio.
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v2 as components
import yaml
from code_editor import code_editor
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors

from . import jobs
from .config import (defaults, dump_settings, effective_parameters, import_config,
                     preset, validate_settings, validate_parameters)
from .geometry import coordinates
from .python_config import render_python_config, apply_python_config
from .profiles import profiles, render_script, validate_profile, validate_slurm
from .resources import estimate_resources, destination_space, measured_run_files
from .storage import inspect_array, human_bytes, read_yaml, atomic_yaml
from .trajectories import discover, preview, displayed_atoms

CLICK_JS = Path(__file__).with_name("scan_click.js").read_text()


def replace_settings(settings: dict):
    revision = st.session_state.get("calc_editor_revision", 0) + 1
    for key in list(st.session_state):
        if key.startswith(("edit_", "calc_")) or key in ("advanced_yaml", "profile_yaml", "preview_context"):
            del st.session_state[key]
    st.session_state["calc_editor_revision"] = revision
    st.session_state.settings = settings
    st.session_state.last_input_folder = settings["input_folder"]


def apply_bead_order(order, selected):
    if isinstance(order, list) and sorted(order) == sorted(selected):
        st.session_state[st.session_state["input_selection_key"]] = order
        st.session_state.settings["inputs"]["files"] = order
    else:
        st.session_state["bead_order_error"] = "The list must contain each selected file exactly once."


def change_folder(path: str):
    st.session_state.settings["input_folder"] = path
    st.session_state.settings["inputs"]["files"] = []
    st.session_state["edit_folder"] = path
    st.session_state.pop("edit_files", None)


def error_key(key: str, message: str | None):
    errors = st.session_state.setdefault("draft_errors", {})
    if message:
        errors[key] = message
        st.error(message)
    else:
        errors.pop(key, None)


def yaml_field(label: str, value, key: str, help=None):
    raw = st.text_area(label, value=yaml.safe_dump(value, default_flow_style=True).strip(),
                       key="edit_"+key, help=help, height=90)
    try:
        parsed = yaml.safe_load(raw)
        error_key(key, None)
        return parsed
    except yaml.YAMLError as exc:
        error_key(key, f"{label}: {exc}")
        return value


def sync_scalar(mapping, key, widget_key):
    mapping[key] = st.session_state[widget_key]


def number(label, mapping, key, owner, *, integer=False, min_value=None, help=None):
    value = mapping[key]
    widget_key = f"edit_{owner}_{key}"
    if integer:
        value = int(value)
    else:
        value = float(value)
    mapping[key] = st.number_input(label, value=value, min_value=min_value,
                                  step=1 if integer else 0.1, key=widget_key, help=help,
                                  on_change=sync_scalar, args=(mapping, key, widget_key))
    return mapping[key]


def vec(label, mapping, key, owner, *, integer=False, min_value=None):
    values = list(mapping[key])
    cols = st.columns(len(values))
    for i, col in enumerate(cols):
        with col:
            values[i] = st.number_input(f"{label} {'xyz'[i] if len(values)<=3 else i}",
                                       value=int(values[i]) if integer else float(values[i]),
                                       step=1 if integer else 0.1, min_value=min_value,
                                       key=f"edit_{owner}_{key}_{i}")
    mapping[key] = values


def choice(label, mapping, key, options, owner, help=None):
    if mapping[key] not in options:
        options = [mapping[key]] + list(options)
    mapping[key] = st.selectbox(label, options, index=options.index(mapping[key]),
                                key=f"edit_{owner}_{key}", help=help)
    return mapping[key]


def textbox(label, mapping, key, owner, help=None):
    mapping[key] = st.text_input(label, value=str(mapping.get(key, "")),
                                key=f"edit_{owner}_{key}", help=help)


def sidebar(s: dict):
    with st.sidebar:
        st.title("Torched TACAW Studio")
        st.caption("Prepare trajectories · define scans · run TACAW")
        textbox("Input folder", s, "input_folder", "input")
        # Keep one stable public key for path changes made by the browser.
        if st.session_state.get("edit_folder") is not None:
            s["input_folder"] = st.session_state["edit_folder"]
            del st.session_state["edit_folder"]
            st.session_state.pop("edit_input_input_folder", None)
            st.rerun()
        previous = st.session_state.get("last_input_folder", s["input_folder"])
        if previous != s["input_folder"]:
            s["inputs"]["files"] = []
            s["view"] = {}
            for key in ("edit_files", "edit_bead_order", "edit_preview_file", "edit_preview_frame", "edit_z_min", "edit_z_max", "preview_context"):
                st.session_state.pop(key, None)
        st.session_state.last_input_folder = s["input_folder"]
        with st.expander("Browse folders"):
            folder = Path(s["input_folder"]).expanduser()
            if folder.is_dir():
                st.button("Parent folder", on_click=change_folder, args=(str(folder.parent),))
                options = sorted([p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")])[:200]
                if options:
                    selected = st.selectbox("Subfolder", options)
                    st.button("Open subfolder", on_click=change_folder, args=(str(folder/selected),))
        workspace_root = Path(__file__).resolve().parents[3]
        example_root = workspace_root/"torched_TACAW"
        if example_root.is_dir():
            with st.expander("Example input folders"):
                for name in ("SiC", "Si_Langevin_300K", "BTO_Ba-O_bussi", "STO_Ti-O_bussi"):
                    st.button(name, on_click=change_folder, args=(str(example_root/name),))
        with st.expander("Calculation presets"):
            name = st.selectbox("Preset", ["SiC plane wave", "Si line scan", "BTO 2D scan", "STO line scan"])
            if st.button("Apply preset"):
                replace_settings(preset(name, s))
                st.rerun()
        with st.expander("Save / load settings"):
            uploaded = st.file_uploader("Studio or TACAW YAML", type=["yaml", "yml", "json"])
            if st.button("Load settings", disabled=uploaded is None):
                try:
                    replace_settings(import_config(uploaded.getvalue().decode(), s))
                    st.session_state.draft_errors = {}
                    st.rerun()
                except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
                    st.error(str(exc))
            save_slot = st.empty()
    return save_slot


def inputs_view(s: dict):
    inputs = s["inputs"]
    st.subheader("Trajectory inputs")
    inputs["recursive"] = st.checkbox("Search subfolders", value=inputs["recursive"], key="edit_recursive")
    inputs["bead_set"] = st.checkbox("Synchronized bead set", value=inputs["bead_set"], key="edit_beads",
        help="Selected files form one calculation. They must have matching frames, atom ordering, cells, and time indices.")
    try:
        found = [str(p) for p in discover(s["input_folder"], inputs["recursive"])]
        st.caption(f"Found {sum(Path(p).suffix == '.traj' for p in found)} ASE trajectories and {sum(Path(p).suffix != '.traj' for p in found)} convertible sources.")
        options = list(dict.fromkeys(found + inputs["files"]))
        selected = inputs["files"] or (options[:1] if options else [])
        selection_key = "edit_files_" + hashlib.sha256(str(Path(s["input_folder"]).expanduser().absolute()).encode()).hexdigest()[:12]
        st.session_state["input_selection_key"] = selection_key
        if selection_key not in st.session_state:
            st.session_state[selection_key] = selected
        inputs["files"] = st.multiselect("Trajectory files (ordered bead list)", options, key=selection_key)
        if inputs["files"]:
            st.dataframe(pd.DataFrame({"Bead": list(range(len(inputs["files"]))),
                                       "Source": inputs["files"]}), hide_index=True)
            if len(inputs["files"]) > 1:
                order = yaml_field("Explicit bead order (paths)", inputs["files"], "bead_order",
                                   help="Reorder paths here and click Apply bead order.")
                st.button("Apply bead order", on_click=apply_bead_order, args=(order, inputs["files"]))
                if "bead_order_error" in st.session_state:
                    st.error(st.session_state.pop("bead_order_error"))
        else:
            st.info("Choose an input folder containing a trajectory.")
    except (OSError, ValueError) as exc:
        st.error(str(exc))
    with st.expander("Trajectory conversion / frame selection", expanded=False):
        c = inputs["conversion"]
        st.caption("Defaults preserve all frames. Existing .traj files are referenced unless you request a subset or metadata override.")
        choice("Source format", c, "format", ["auto", "xyz", "extxyz", "ipi", "lammps-dump-text"], "conversion")
        cols = st.columns(3)
        with cols[0]:
            number("Start frame", c, "start", "conversion", integer=True, min_value=0)
        with cols[1]:
            unlimited = st.checkbox("To end of trajectory", value=c["stop"] is None, key="edit_to_end")
            if unlimited:
                c["stop"] = None
            else:
                c["stop"] = st.number_input("Stop frame (exclusive)", value=c["stop"] or max(c["start"]+1, 3800),
                                            min_value=1, key="edit_conversion_stop")
        with cols[2]:
            number("Conversion stride", c, "stride", "conversion", integer=True, min_value=1)
        cols = st.columns(2)
        with cols[0]:
            choice("Position units", c, "positions_unit", ["auto", "angstrom", "bohr"], "conversion")
        with cols[1]:
            choice("Cell units", c, "cell_unit", ["auto", "angstrom", "bohr"], "conversion")
        c["cell"] = yaml_field("Cell override in Å (null = use metadata)", c["cell"], "cell",
                              help="Three lengths, six cell parameters, or a 3×3 matrix. The current TACAW engine needs a diagonal cell.")
        c["pbc"] = yaml_field("PBC override (null = use metadata)", c["pbc"], "pbc",
                              help="Example: [true, true, true].")
        c["type_symbols"] = yaml_field("LAMMPS type → element mapping", c["type_symbols"], "types",
                                      help="Example: {1: Ba, 2: Ti, 3: O}. Required when the dump does not identify elements.")
        st.caption("Effective prepared timestep = source timestep × conversion stride. Chunk step is applied separately.")


@st.cache_data(max_entries=4, show_spinner=False)
def cached_preview(path, index, options_text, size, mtime):
    return preview(path, index, yaml.safe_load(options_text))


def click_callback():
    event = st.session_state.get("scan_click", {}).get("clicked")
    context = st.session_state.get("preview_context")
    if not event or not context or event.get("preview_id") != context["id"]:
        return
    point = np.array([event["x"], event["y"]], dtype=float)
    if not np.isfinite(point).all():
        return
    if context["snap"] and len(context["positions"]):
        xy = context["positions"][:, :2]
        point = xy[np.argmin(np.sum((xy-point)**2, axis=1))]
    scan = st.session_state.settings["scan"]
    role = event["role"]
    if role == "Start / origin":
        scan["start"] = point.tolist()
        for i in range(2):
            st.session_state[f"edit_scan_start_{i}"] = float(point[i])
    elif role == "End":
        scan["end"] = point.tolist()
        for i in range(2):
            st.session_state[f"edit_scan_end_{i}"] = float(point[i])
    else:
        axis = 0 if role == "Vector 1 corner" else 1
        scan["basis"][axis] = (point-np.array(scan["start"])).tolist()
        for i in range(2):
            st.session_state[f"edit_scan_basis_{axis}_{i}"] = float(scan["basis"][axis][i])


def trajectory_figure(positions, numbers, indices, grid, labels=False, revision=""):
    fig = go.Figure()
    for z in sorted(set(numbers.tolist())):
        mask = numbers == z
        color = "rgb("+",".join(str(int(v*255)) for v in jmol_colors[z])+")"
        fig.add_trace(go.Scattergl(
            x=positions[mask, 0].tolist(), y=positions[mask, 1].tolist(),
            mode="markers+text" if labels else "markers", name=chemical_symbols[z],
            text=[chemical_symbols[z]]*int(mask.sum()), textposition="top center",
            customdata=indices[mask].tolist(),
            marker=dict(color=color, size=9),
            hovertemplate=chemical_symbols[z]+" · atom %{customdata}<br>x=%{x:.5f} Å<br>y=%{y:.5f} Å<extra></extra>"))
    if grid is not None:
        pts = grid.reshape(-1, 2)
        fig.add_trace(go.Scatter(x=pts[:, 0].tolist(), y=pts[:, 1].tolist(),
                                mode="lines+markers" if grid.shape[1] == 1 else "markers",
                                name="Probe positions", marker=dict(color="#dc2626", size=5),
                                hovertemplate="Probe x=%{x:.5f} Å<br>y=%{y:.5f} Å<extra></extra>"))
        fig.add_trace(go.Scatter(x=[pts[0, 0], pts[-1, 0]], y=[pts[0, 1], pts[-1, 1]],
                                mode="markers", name="First / last probe",
                                marker=dict(color=["#16a34a", "#7c3aed"], size=12, symbol="x")))
    fig.update_layout(height=650, xaxis_title="x (Å)", yaxis_title="y (Å)",
                      dragmode="pan", uirevision=revision, margin=dict(l=30, r=10, t=20, b=30),
                      legend=dict(orientation="h"), hovermode="closest")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def scan_view(s: dict):
    st.subheader("Scan geometry")
    scan_col, plot_col = st.columns([1, 2], gap="large")
    with scan_col:
        with st.expander("Scan settings", expanded=True):
            p = geometry_editor(s)
    with plot_col:
        # Populate the plot after reading its collapsed display controls so
        # widget edits update the overlay in the same rerun.
        plot_area = st.container()
        preview_col, display_col = st.columns(2, gap="large")
        with preview_col:
            preview_controls = st.expander("Trajectory preview settings")
        with display_col:
            display_controls = st.expander("Atom display settings")
    trajectory_view(s, p, plot_area, preview_controls, display_controls)


def change_scan_mode(scan: dict):
    scan["kind"] = st.session_state["edit_scan_kind"]
    if scan["kind"] == "2D scan":
        # A line's inactive axis has one point and often zero extent. Seed an
        # editable rectangle only on an explicit mode change, not on every edit.
        span = max(abs(float(scan["end"][i]) - float(scan["start"][i])) for i in range(2)) or 1.
        for axis in range(2):
            if np.isclose(scan["end"][axis], scan["start"][axis]):
                scan["end"][axis] = float(scan["start"][axis]) + span
            if scan["shape"][axis] < 2:
                scan["shape"][axis] = 2
                scan["batch"][axis] = 1
            st.session_state[f"edit_scan_end_{axis}"] = float(scan["end"][axis])
            st.session_state[f"edit_scan_shape_{axis}"] = int(scan["shape"][axis])
            st.session_state[f"edit_scan_batch_{axis}"] = int(scan["batch"][axis])


def geometry_editor(s: dict):
    scan = s["scan"]
    kinds = ["Plane wave", "Line scan", "2D scan", "Unit cell", "Basis vectors"]
    if scan["kind"] == "Native box":
        kinds.append("Native box")
    scan["kind"] = st.selectbox("Calculation mode", kinds, index=kinds.index(scan["kind"]),
                                key="edit_scan_kind", on_change=change_scan_mode, args=(scan,))
    if scan["kind"] == "2D scan":
        st.caption("Start and End are opposite rectangle corners: both x and y must differ. "
                   "Each axis needs at least 2 scan points. Switching from a line creates an editable rectangle.")
    if scan["kind"] in ("Line scan", "2D scan", "Basis vectors"):
        vec("Start / origin (Å)", scan, "start", "scan")
        if scan["kind"] != "Basis vectors":
            vec("End (Å)", scan, "end", "scan")
            scan["periodic"] = st.checkbox("Periodic sampling: exclude final edge", value=scan["periodic"], key="edit_periodic")
        else:
            for axis in range(2):
                for i, col in enumerate(st.columns(2)):
                    with col:
                        scan["basis"][axis][i] = st.number_input(
                            f"Basis vector {axis+1} {'xy'[i]} (Å)", value=float(scan["basis"][axis][i]),
                            key=f"edit_scan_basis_{axis}_{i}")
    if scan["kind"] not in ("Plane wave", "Native box"):
        if scan["kind"] == "Line scan":
            scan["shape"][0] = st.number_input("Line points", value=int(scan["shape"][0]), min_value=2, key="edit_line_points")
            scan["batch"][0] = st.number_input("Points per batch", value=int(scan["batch"][0]), min_value=1, key="edit_line_batch")
            scan["shape"][1], scan["batch"][1] = 1, 1
        else:
            vec("Scan points", scan, "shape", "scan", integer=True, min_value=1)
            vec("Batch points", scan, "batch", "scan", integer=True, min_value=1)
    if scan["kind"] == "Native box":
        st.info("Imported native box geometry is editable in Calculation → Config editor.")
    try:
        p = effective_parameters(s)
        validate_parameters(p)
        error_key("scan", None)
    except (ValueError, TypeError, KeyError) as exc:
        error_key("scan", str(exc))
        p = None
    return p


def trajectory_view(s: dict, p, plot_area, preview_controls, display_controls):
    scan = s["scan"]
    files = s["inputs"]["files"]
    if not files:
        with plot_area:
            st.info("Select a trajectory to show atoms and the scan.")
        return
    view = s.setdefault("view", {})
    with preview_controls:
        preview_file = st.selectbox("Preview trajectory", files, index=min(view.get("bead", 0), len(files)-1), key="edit_preview_file")
        view["bead"] = files.index(preview_file)
        index = st.number_input("Preview frame", value=int(view.get("frame", 0)), step=1, key="edit_preview_frame")
        if st.button("Load frame"):
            view["frame"] = int(index)
    try:
        path = Path(preview_file).expanduser()
        options_text = yaml.safe_dump(s["inputs"]["conversion"])
        with plot_area:
            with st.spinner("Reading one trajectory frame…"):
                atoms, count = cached_preview(str(path), int(view.get("frame", 0)), options_text,
                                              path.stat().st_size, path.stat().st_mtime_ns)
        with preview_controls:
            lengths = ", ".join(f"{length:.6g}" for length in atoms.cell.lengths())
            st.caption(f"Cell lengths (Å): {lengths}")
        with display_controls:
            extent = float(atoms.cell.lengths()[2]) or float(np.max(atoms.positions[:, 2])+1)
            cols = st.columns(2)
            with cols[0]:
                view["z_min"] = st.number_input("Display z minimum (Å)", value=float(view.get("z_min", 0.)), key="edit_z_min")
            with cols[1]:
                view["z_max"] = st.number_input("Display z maximum (Å)", value=float(view.get("z_max", extent+1e-7)), key="edit_z_max")
            one_repeat = st.checkbox("Display one z repeat", value=view.get("one_repeat", False), key="edit_one_repeat")
            view["one_repeat"] = one_repeat
            repeats = st.number_input("Original z repeats", value=int(view.get("repeats", 1)), min_value=1,
                                      key="edit_z_repeats", disabled=not one_repeat)
            view["repeats"] = repeats
            if p is not None and p.get("center_atoms_in_cell"):
                first, _ = cached_preview(files[0], s["inputs"]["conversion"]["start"], options_text,
                                          Path(files[0]).stat().st_size, Path(files[0]).stat().st_mtime_ns)
                centered = first.copy()
                centered.center()
                atoms = atoms.copy()
                atoms.translate(centered.get_center_of_mass()-first.get_center_of_mass())
                st.caption("Atoms are displayed with the same first-frame centering shift used by TACAW.")
            view["labels"] = st.checkbox("Show element labels on atoms", value=view.get("labels", False), key="edit_atom_labels")
        positions, numbers, indices = displayed_atoms(atoms, view["z_min"], view["z_max"], repeats if one_repeat else None)
        grid = None
        if p is not None and math.prod(p["scanning_shape"]) <= 100000:
            # Structure overrides also control engine unit-cell geometry.
            cell = atoms.cell.array
            if p.get("sample_structure_file"):
                from ase.io import read
                cell = read(Path(p["sample_structure_file"]).expanduser()).cell.array
            grid = coordinates(p, cell)
        preview_id = hashlib.sha256((str(path)+str(path.stat().st_mtime_ns)+str(view.get("frame", 0))+options_text).encode()).hexdigest()
        with plot_area:
            st.caption(f"Frame {view.get('frame', 0)} · {len(atoms):,} atoms" + (f" · {count:,} total frames" if count is not None else " · streaming source"))
            click_col, snap_col = st.columns([2, 1])
            with click_col:
                roles = ["Start / origin", "Vector 1 corner", "Vector 2 corner"] if scan["kind"] == "Basis vectors" else ["Start / origin", "End"]
                role = st.radio("Click sets", roles, horizontal=True, key="edit_click_role")
            with snap_col:
                view["snap"] = st.checkbox("Snap selected coordinates to nearest displayed atom", value=view.get("snap", False), key="edit_snap")
            enabled = scan["kind"] in ("Line scan", "2D scan", "Basis vectors")
            st.caption(f"Selection target: {role}. Click anywhere in the plot. Drag to pan; scroll to zoom. Atom indices start at zero.")
            st.session_state.preview_context = {"id": preview_id, "positions": positions, "snap": view["snap"]}
            st.plotly_chart(trajectory_figure(positions, numbers, indices, grid, view["labels"], preview_id),
                            key="trajectory_plot", config={"scrollZoom": True, "displaylogo": False})
            bridge = components.component("tacaw_scan_click", js=CLICK_JS)
            bridge(data={"enabled": enabled, "role": role, "preview_id": preview_id},
                   key="scan_click", on_clicked_change=click_callback)
            if grid is not None:
                st.caption(f"{math.prod(grid.shape[:2]):,} probes · first {grid[0,0].round(6).tolist()} Å · last {grid[-1,-1].round(6).tolist()} Å")
                st.download_button("Download probe coordinates", _coordinates_csv(grid), "scan-coordinates.csv", "text/csv")
            elif p is not None:
                st.info("Probe overlay is disabled above 100,000 positions; the calculation keeps the complete grid.")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        with plot_area:
            st.error(f"Trajectory preview: {exc}")


def _coordinates_csv(grid):
    from io import StringIO
    buf = StringIO()
    rows = np.column_stack([np.indices(grid.shape[:2]).reshape(2, -1).T, grid.reshape(-1, 2)])
    np.savetxt(buf, rows, delimiter=",", header="probe_i,probe_j,x_A,y_A", comments="")
    return buf.getvalue()


def calculation_view(s: dict):
    st.subheader("Calculation configuration")
    st.caption("Edit Config values and click Apply configuration inside the editor, or press Ctrl/Cmd+Enter. "
               "Use actual values instead of args.*; simple variables and arithmetic also work.")
    canonical = render_python_config(s)
    if "calc_text" not in st.session_state:
        st.session_state.calc_text = canonical
        st.session_state.calc_applied_text = canonical
        st.session_state.calc_settings_text = canonical
    if canonical != st.session_state.calc_settings_text:
        if st.session_state.calc_text == st.session_state.calc_applied_text:
            st.session_state.calc_text = canonical
            st.session_state.calc_applied_text = canonical
            st.session_state.calc_editor_revision = st.session_state.get("calc_editor_revision", 0) + 1
        else:
            st.session_state.calc_external_change = True
        st.session_state.calc_settings_text = canonical
    buttons = [{"name": "Apply configuration", "feather": "Check", "primary": True,
                "hasText": True, "alwaysOn": True, "commands": ["submit"],
                "style": {"top": "0rem", "right": "0.5rem"}}]
    revision = st.session_state.get("calc_editor_revision", 0)
    response = code_editor(st.session_state.calc_text, lang="python", theme="default",
                           height="680px", buttons=buttons, response_mode="blur", allow_reset=True,
                           options={"wrap": False, "showLineNumbers": True, "tabSize": 4,
                                    "useSoftTabs": True, "fontSize": 15, "showPrintMargin": False},
                           key=f"calc_editor_{revision}")
    if response.get("id") and response["id"] != st.session_state.get("calc_event_id"):
        st.session_state.calc_event_id = response["id"]
        st.session_state.calc_text = response["text"]
        if response.get("type") != "submit" and response["text"] == st.session_state.calc_applied_text:
            st.session_state.pop("calc_error", None)
        if response.get("type") == "submit":
            try:
                updated = apply_python_config(s, response["text"])
                raw = response["text"]
                replace_settings(updated)
                st.session_state.calc_text = raw
                st.session_state.calc_applied_text = raw
                st.session_state.calc_settings_text = render_python_config(updated)
                st.session_state.calc_notice = "Configuration applied. Scan, inputs and resource estimates updated."
                st.session_state.draft_errors = {}
                st.rerun()
            except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError) as exc:
                st.session_state.calc_error = str(exc)
    pending = st.session_state.calc_text != st.session_state.calc_applied_text
    errors = st.session_state.setdefault("draft_errors", {})
    if pending:
        errors["calculation_editor"] = "Apply your Python configuration before generating a run."
        st.caption("Unapplied edits. Click Apply configuration to validate and use them.")
    else:
        errors.pop("calculation_editor", None)
    if st.session_state.get("calc_error"):
        st.error(st.session_state.calc_error)
    if st.session_state.get("calc_external_change"):
        st.info("Scan or input settings changed while you had draft code. Reload current settings to use them, "
                "or apply your draft to use the values written in the editor.")
    if st.session_state.get("calc_notice") and not pending and not st.session_state.get("calc_error"):
        st.success(st.session_state.calc_notice)
    controls = st.columns(3)
    with controls[0]:
        if st.button("Reload current settings"):
            st.session_state.calc_text = canonical
            st.session_state.calc_applied_text = canonical
            st.session_state.calc_settings_text = canonical
            st.session_state.calc_editor_revision = revision + 1
            for key in ("calc_error", "calc_external_change", "calc_notice"):
                st.session_state.pop(key, None)
            errors.pop("calculation_editor", None)
            st.rerun()
    with controls[1]:
        st.download_button("Download applied Python config", canonical, "tacaw-settings.py", "text/x-python")
    with controls[2]:
        try:
            st.download_button("Download applied TACAW YAML", yaml.safe_dump(effective_parameters(s), sort_keys=False),
                               "tacaw-inputs.yaml", "application/yaml")
        except (ValueError, KeyError, TypeError):
            pass
    st.caption("datafolder is the parent folder for new runs; intensities_zarray is a filename inside each run. "
               "config_file, derived axes and batch status are generated automatically. "
               "This editor reads settings without executing a calculation.")
    try:
        validate_parameters(effective_parameters(s))
        error_key("parameters", None)
    except (ValueError, TypeError, KeyError) as exc:
        error_key("parameters", str(exc))


def machine_view(s: dict):
    st.subheader("Calculation environment")
    execution = s["execution"]
    builtins = profiles()
    saved_path = Path(__file__).resolve().parents[1]/".studio_profiles.yaml"
    try:
        if saved_path.exists():
            builtins.update(read_yaml(saved_path))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        st.warning(f"Could not read saved profiles: {exc}")
    selected = st.selectbox("Profile template", list(builtins), key="profile_template")
    if st.button("Load profile template"):
        execution["profile"] = copy.deepcopy(builtins[selected])
        for key in list(st.session_state):
            if key.startswith("edit_profile_") or key == "profile_yaml":
                del st.session_state[key]
        st.rerun()
    profile = execution["profile"]
    textbox("Profile name", profile, "name", "profile")
    choice("Scheduler", profile, "scheduler", ["local", "slurm"], "profile")
    choice("Torch backend", profile, "backend", ["cpu", "cuda", "rocm"], "profile")
    for key, label in (("python", "Simulation Python command (list)"), ("launcher", "Container / launcher command (list)"),
                       ("modules", "Modules to load"), ("pythonpath", "Python dependency paths"),
                       ("environment", "Environment variables"), ("srun_args", "Additional srun arguments")):
        profile[key] = yaml_field(label, profile[key], "profile_"+key)
    choice("Worker device binding", profile, "worker_binding", ["scheduler", "local-rank", "rocr-local-rank"], "profile",
           help="scheduler: one visible GPU per task. local-rank: all GPUs visible, use cuda:local_rank. rocr-local-rank: restrict an AMD worker to its local-rank GPU.")
    profile["setup_commands"] = st.text_area("Additional environment setup commands", value=profile["setup_commands"], key="edit_profile_setup",
                                           help="Explicit shell commands executed by launchers, e.g. source a module initialization script or activate an environment.")
    if profile["name"] == "Workspace CPU":
        st.caption("Uses the existing workspace Torch/pyms environment without modifying it.")
    if profile["name"] in ("LUMI", "Dardel"):
        st.info("Cluster templates require checking module/container availability and setting your account, partition, and simulation dependency paths. See docs/setup.md.")
    try:
        validate_profile(profile)
        error_key("profile", None)
    except (ValueError, TypeError, KeyError) as exc:
        error_key("profile", str(exc))
    if st.button("Save profile on this computer"):
        try:
            validate_profile(profile)
            saved = read_yaml(saved_path) if saved_path.exists() else {}
            saved[profile["name"]] = copy.deepcopy(profile)
            atomic_yaml(saved_path, saved)
            st.success("Profile saved.")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    st.download_button("Download machine profile", yaml.safe_dump(profile, sort_keys=False),
                       "tacaw-profile.yaml", "application/yaml")
    st.subheader("SBATCH settings")
    slurm = execution["slurm"]
    cols = st.columns(3)
    for i, (key, label) in enumerate((("account", "Account"), ("partition", "Partition"), ("qos", "QoS"),
                                    ("time", "Time limit"), ("memory", "Memory (e.g. 120G)"),
                                    ("output", "Slurm output path"), ("error", "Slurm error path"))):
        with cols[i % 3]:
            textbox(label, slurm, key, "slurm")
    cols = st.columns(4)
    for col, (key, label, minimum) in zip(cols, (("nodes", "Nodes", 1), ("tasks_per_node", "Tasks per node", 1),
                                               ("cpus_per_task", "CPUs per task", 1), ("gpus_per_node", "GPUs per node", 0))):
        with col:
            number(label, slurm, key, "slurm", integer=True, min_value=minimum)
    slurm["additional"] = yaml_field("Additional SBATCH directives", slurm["additional"], "slurm_extra",
                                     help="Example: {constraint: gpu, exclusive: null}. Null adds a flag without a value.")
    try:
        validate_slurm(slurm, profile["backend"])
        error_key("slurm", None)
    except (ValueError, TypeError, KeyError) as exc:
        error_key("slurm", str(exc))


def resource_estimate_view(s: dict, folder: Path | None = None):
    title = "Resources for saved run" if folder else "Estimated resources for new run"
    with st.expander(title, expanded=True):
        try:
            config = None
            info = None
            if folder and (folder / "scan_gpu_config.yaml").is_file():
                config = read_yaml(folder / "scan_gpu_config.yaml")
                array_path = Path(config["storage"]["intensities_zarray"])
                if array_path.exists():
                    info = inspect_array(array_path)
            lengths = atoms_count = elements_count = source_frames = None
            files = s["inputs"]["files"]
            if files:
                path = Path(files[0]).expanduser()
                if not path.is_absolute():
                    path = Path(s["input_folder"]).expanduser() / path
                options_text = yaml.safe_dump(s["inputs"]["conversion"])
                try:
                    atoms, source_frames = cached_preview(str(path), s["inputs"]["conversion"]["start"],
                                                          options_text, path.stat().st_size, path.stat().st_mtime_ns)
                    lengths = atoms.cell.lengths()
                    atoms_count, elements_count = len(atoms), len(set(atoms.numbers.tolist()))
                except (OSError, ValueError, KeyError, TypeError, IndexError):
                    if config is None:
                        raise
                    # Saved array dimensions remain useful when an external
                    # source is offline; atom metadata is optional for sizing.
                    if (folder / "structure.extxyz").is_file():
                        from ase.io import read
                        atoms = read(folder / "structure.extxyz")
                        lengths = atoms.cell.lengths()
                        atoms_count, elements_count = len(atoms), len(set(atoms.numbers.tolist()))
            structure_file = s["parameters"].get("sample_structure_file")
            if structure_file and config is None:
                from ase.io import read
                path = Path(structure_file).expanduser()
                if not path.is_absolute():
                    path = Path(s["input_folder"]).expanduser() / path
                lengths = read(path).cell.lengths()
            report = estimate_resources(s, cell_lengths=lengths, atom_count=atoms_count,
                                        element_count=elements_count, source_frames=source_frames,
                                        engine_config=config, array_info=info)
            cols = st.columns(4)
            labels = [("Zarr data (uncompressed)", report["result_bytes"]),
                      ("NumPy export (extra)", report["npy_bytes"]),
                      ("RAM / worker (approx.)", report["ram_plan_bytes"]),
                      ("GPU VRAM / worker (approx.)", report["gpu_plan_bytes"])]
            for col, (label, value) in zip(cols, labels):
                with col:
                    st.metric(label, "Not used (CPU)" if "GPU" in label and not value else human_bytes(value))
            st.caption("GiB = 1024³ bytes. Memory includes estimated headroom; it is not a measured peak or a guaranteed limit. "
                       "Zarr compression and metadata change its actual disk size. NumPy needs a separate full-size file.")
            state = "Saved engine/array dimensions" if report["prepared"] else "Preview dimensions; recalculated when the run is prepared"
            st.caption(f"{state} · result shape {report['shape']} · {report['dtype']} · "
                       f"{report['frequency_bins']:,} energy bins · {report['worker_count']} worker(s)")
            destination = folder or Path(s.get("output_root") or Path(s["input_folder"]) / "tacaw_runs")
            space = destination_space(destination)
            with st.expander("Size and memory breakdown"):
                rows = [
                    {"Item": "One decoded output chunk", "Size": human_bytes(report["chunk_bytes"])},
                    {"Item": "Retained complex wave array / worker", "Size": human_bytes(report["wave_bytes"])},
                    {"Item": "Modeled FFT working arrays / worker", "Size": human_bytes(report["fft_bytes"])},
                    {"Item": "Modeled multislice working arrays / worker", "Size": human_bytes(report["multislice_bytes"])},
                    {"Item": "Known RAM arrays / worker", "Size": human_bytes(report["ram_arrays_bytes"])},
                    {"Item": "Known GPU arrays / worker", "Size": human_bytes(report["gpu_arrays_bytes"])},
                    {"Item": f"Planning RAM / node ({report['workers_per_node']} workers)", "Size": human_bytes(report["ram_node_plan_bytes"])},
                    {"Item": "Zarr + optional NumPy (uncompressed data)", "Size": human_bytes(report["result_bytes"] + report["npy_bytes"])},
                    {"Item": "Converted trajectories (positions only)", "Size": "Unknown until conversion" if report["converted_bytes"] is None else human_bytes(report["converted_bytes"])},
                    {"Item": "Free filesystem space (quota not checked)", "Size": human_bytes(space["free_bytes"])},
                ]
                st.table(rows)
                st.caption(f"Filesystem checked at {space['path']}. "
                           + ("Automatic NumPy export is enabled." if s["auto_npy"] else "Automatic NumPy export is disabled; its extra size is optional."))
                st.caption("Planning memory = 1.5 × modeled working arrays + 1 GiB per worker. "
                           "CPU runs keep simulation arrays in RAM. GPU runs keep them in VRAM and retain host buffers in RAM. "
                           "Torch, FFT libraries, allocators and codecs may need more. Converted trajectory headers and extra arrays are additional.")
                st.caption("Reduce points per batch, frames per TACAW chunk, grid size, or multislice slices to lower memory. "
                           "Reduce total scan points, output ROI or frequency ROI to lower result size. "
                           "Trajectory chunks are averaged into one result; they do not create separate output copies.")
                if not report["element_count_known"]:
                    st.caption("No atom metadata is loaded; the potential estimate assumes one element.")
            if folder:
                env_file = folder / "environment.json"
                if env_file.is_file():
                    environment = json.loads(env_file.read_text())
                    devices = environment.get("gpu_devices", [])
                    node_ram = environment.get("node_memory", {}).get("total_bytes")
                    if node_ram:
                        st.caption(f"Environment check: physical RAM on that node {human_bytes(node_ram)}; the Slurm memory allocation may be smaller.")
                    if devices:
                        st.caption("GPUs visible during the environment check: " + "; ".join(
                            f"{device['name']}: {human_bytes(device['total_bytes'])}" for device in devices))
                if st.button("Measure files on disk", key=f"measure_files_{folder}"):
                    with st.spinner("Measuring run files without reading array data…"):
                        st.session_state[f"run_file_sizes_{folder}"] = measured_run_files(folder)
                measured = st.session_state.get(f"run_file_sizes_{folder}")
                if measured:
                    st.write(f"Measured run files: **{human_bytes(measured['total_bytes'])}**")
                    st.table([{"Files": name, "Size on disk": human_bytes(size)}
                              for name, size in measured["groups"].items()])
                    st.caption("Last manual measurement; press Measure files on disk to update. External referenced inputs are excluded.")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            st.info(f"Resource estimate unavailable: {exc}")


def run_view(s: dict):
    st.subheader("Generate and launch")
    textbox("Run name", s, "run_name", "output")
    textbox("Output root (blank = input folder / tacaw_runs)", s, "output_root", "output")
    textbox("Intensity Zarr basename", s, "intensities_name", "output")
    s["auto_npy"] = st.checkbox("Convert completed results to NumPy automatically", value=s["auto_npy"], key="edit_auto_npy")
    st.caption("Generating a folder saves scripts and settings. Conversion and calculation start only when you launch it.")
    resource_estimate_view(s)
    try:
        validate_settings(s)
        valid = not st.session_state.get("draft_errors")
        example_folder = Path(s.get("output_root") or Path(s["input_folder"])/"tacaw_runs")/"<new-run>"
        with st.expander("Preview launch script", expanded=False):
            st.code(render_script(s, example_folder, slurm=s["execution"]["profile"]["scheduler"] == "slurm"), language="bash")
    except (ValueError, TypeError, KeyError) as exc:
        valid = False
        st.error(str(exc))
    if st.session_state.get("draft_errors"):
        if st.session_state.draft_errors.get("calculation_editor"):
            st.warning("Apply your Python configuration in Calculation before generating a run.")
        else:
            st.warning("Correct invalid settings before generating a run.")
    if st.button("Generate run folder", disabled=not valid or not s["inputs"]["files"], type="primary"):
        try:
            folder = jobs.generate_run(copy.deepcopy(s))
            st.session_state["edit_run_folder"] = str(folder)
            st.success(f"Generated {folder}")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    st.divider()
    run_folder = st.text_input("Existing / generated run folder", key="edit_run_folder")
    if not run_folder:
        return
    folder = Path(run_folder).expanduser().absolute()
    if not (folder/"studio.yaml").is_file():
        st.error("This folder does not contain studio.yaml.")
        return
    try:
        frozen = read_yaml(folder/"studio.yaml")
        st.caption(f"Frozen run settings · {frozen['execution']['profile']['name']} · {folder}")
        resource_estimate_view(frozen, folder)
        with st.expander("Saved settings and scripts"):
            st.code((folder/"studio.yaml").read_text(), language="yaml")
            script = "submit.sh" if frozen["execution"]["profile"]["scheduler"] == "slurm" else "run_local.sh"
            st.code((folder/script).read_text(), language="bash")
            if st.button("Load this run's settings into editor"):
                replace_settings(frozen)
                st.rerun()
        state = jobs.status(folder)
        st.write(f"Job: **{state['state']}** · calculation: **{state['lifecycle'].get('state', 'generated')}**")
        if state["job"].get("job_id"):
            st.write("Slurm job ID:", state["job"]["job_id"])
        if state["lifecycle"].get("error"):
            st.error(state["lifecycle"]["error"])
        if "batches" in state:
            b = state["batches"]
            st.progress(b["finished"]/max(b["total"], 1),
                        text=f"{b['finished']}/{b['total']} batches finished · {b['running']} running · {b['waiting']} waiting")
        cols = st.columns(5)
        actions = [("Run / submit", "run", False), ("Resume", "run", True),
                   ("Check environment", "check", False), ("Convert to NumPy", "convert", False)]
        for col, (label, action, resume) in zip(cols, actions):
            with col:
                if st.button(label, disabled=state["active"]):
                    try:
                        jobs.start(folder, action, resume=resume)
                        st.rerun()
                    except (OSError, ValueError, KeyError, TimeoutError) as exc:
                        st.error(str(exc))
        with cols[4]:
            if st.button("Cancel job", disabled=not state["active"]):
                try:
                    jobs.cancel(folder)
                    st.rerun()
                except (OSError, ValueError, TimeoutError) as exc:
                    st.error(str(exc))
        st.button("Refresh status and logs")
        if (folder/"scan_gpu_config.yaml").exists():
            c = read_yaml(folder/"scan_gpu_config.yaml")
            with st.expander("Derived engine configuration (read only)"):
                st.json({key: c[key] for key in ("sample", "trajectory", "simulation", "storage")}, expanded=False)
            array_path = Path(c["storage"]["intensities_zarray"])
            if array_path.exists():
                info = inspect_array(array_path)
                st.caption(f"Result shape {info['shape']} · {info['dtype']} · NumPy estimate {human_bytes(info['bytes'])} · decoded chunk {human_bytes(info['chunk_bytes'])}, plus codec buffers")
                st.code(str(array_path))
                npy = array_path.with_suffix(".npy")
                if npy.exists():
                    st.write("NumPy result:", str(npy))
                st.caption("Open the Zarr array or its NumPy copy in EELS Studio. Use the saved trajectory timestep and chunk step for energy calibration.")
        logs = sorted((folder/"logs").glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if logs:
            selected = st.selectbox("Log file", [p.name for p in logs], key="edit_log_choice")
            st.code(jobs.tail(folder/"logs"/selected), language="text")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        st.error(f"Run inspection: {exc}")


def main():
    st.set_page_config(page_title="Torched TACAW Studio", page_icon="🔬", layout="wide")
    if "settings" not in st.session_state:
        st.session_state.settings = defaults()
    s = st.session_state.settings
    save_slot = sidebar(s)
    st.title("Torched TACAW Studio")
    st.caption("Trajectory → scan → calculation settings → machine profile → run")
    tabs = st.tabs(["Inputs", "Scan", "Calculation", "Machine & SBATCH", "Run & logs"])
    with tabs[0]:
        inputs_view(s)
    with tabs[1]:
        scan_view(s)
    with tabs[2]:
        calculation_view(s)
    with tabs[3]:
        machine_view(s)
    with tabs[4]:
        run_view(s)
    save_slot.download_button("Download Studio settings", dump_settings(s),
                              "tacaw-studio.yaml", "application/yaml")
