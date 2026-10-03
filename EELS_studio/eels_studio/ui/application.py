"""Application setup, scan selection, and top-level view routing."""
from pathlib import Path
import re

import numpy as np
import streamlit as st

from eels_studio.ui.folder_browser import render_folder_input
from eels_studio.ui.components.file_order import render_file_order
from eels_studio.ui.cache_layer import _cached_inspect, inspect_scan_cached
from eels_studio.io.scan_sources import discover_scans, resolve_data_directory
from eels_studio.ui.workspace_config import render_config_controls, render_config_download
from eels_studio.ui.scan_map_view import render_scan_map
from eels_studio.ui.spectra import render_spectra

# The default folder remains the directory containing app.py, not the package.
ROOT = Path(__file__).resolve().parents[2]


def suggested_label(path):
    match = re.search(r"(?:^|_)T(\d+(?:\.\d+)?)K(?:_|$)", Path(path).stem)
    return f"{match[1]} K" if match else Path(path).stem


def main():
    # Color for the Detector/Spectrum card panels, matching the app's teal theme.
    PANEL_BORDER = "#dce5ec"

    st.set_page_config(page_title="EELS Studio", page_icon="🔬", layout="wide")
    st.markdown(f"""<style>
    .block-container {{padding-top: 3.5rem; padding-bottom: 2rem;}}
    h1 {{letter-spacing: -0.045em;}}
    [data-testid="stMetric"] {{background: white; border: 1px solid {PANEL_BORDER};
      border-radius: 12px; padding: 14px 18px;}}
    [data-testid="stSidebar"] {{border-right: 1px solid {PANEL_BORDER};}}
    /* Keep the interface fully visible while an input change reruns the app. */
    [data-testid="stElementContainer"], [data-testid="stExpanderDetails"] {{
      opacity: 1 !important; transition: none !important;
    }}

    /* Detector / Spectrum card panels */
    .st-key-detector_panel, .st-key-spectrum_panel {{
      background: white; border: 1px solid {PANEL_BORDER}; border-radius: 14px;
      padding: 18px 20px; box-shadow: 0 2px 8px rgba(20,40,80,0.04);
    }}
    .eels-panel-title {{font-size: 19px; font-weight: 600; color: #172b3a; margin: 0 0 8px 0; text-align: center;}}

    /* Compact Plotly modebar */
    .modebar-container {{background: white !important; border: 1px solid {PANEL_BORDER};
      border-radius: 9px; padding: 2px;}}
    </style>""", unsafe_allow_html=True)


    config_slot = render_config_controls()

    st.caption("STEM · SPECTROSCOPY WORKSPACE")
    st.title("EELS Studio")
    st.write("Choose your scans, position the detector, and explore spectra or a 2D probe map.")
    view = st.radio("Visualization", ["Spectra & detector", "2D scan map"], horizontal=True, key="visualization")

    with st.sidebar:
        st.header("Your scans")
        folder = render_folder_input(ROOT)
        st.button("Refresh files", use_container_width=True, on_click=_cached_inspect.clear,
                 help="Force every file to be re-inspected, in case one changed without its size or modified time changing.")
        try:
            directory = resolve_data_directory(folder)
            available = discover_scans(directory)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
            render_config_download(config_slot)
            st.stop()
        if not available:
            st.info("No NumPy or Zarr scans in this folder. Choose a folder containing .npy files "
                    "or Zarr arrays (.zarr/.zarray), or select a Zarr array directory directly.")
            render_config_download(config_slot)
            st.stop()
        # Inspection reads metadata only. Zarr revisions include nested chunk stats.
        scans, rejected = {}, []
        for path in available:
            try:
                scans[str(path)] = inspect_scan_cached(path)
            except (OSError, ValueError, EOFError) as exc:
                rejected.append(f"{path.name}: {exc}")
        if rejected:
            with st.expander(f"{len(rejected)} unsupported file(s)"):
                for reason in rejected:
                    st.caption(reason)
        options = list(scans)
        if not options:
            st.info("No supported 3D or 6D scan arrays found.")
            render_config_download(config_slot)
            st.stop()
        files_key = f"files:{directory}"
        if files_key in st.session_state:
            missing = [p for p in st.session_state[files_key] if p not in scans]
            if missing:
                st.warning("Saved scans unavailable in this folder: " + ", ".join(Path(p).name for p in missing))
                st.session_state[files_key] = [p for p in st.session_state[files_key] if p in scans]
        selected = st.multiselect("Files to compare", options, default=options[:2],
                                 format_func=lambda p: Path(p).name, key=f"files:{directory}")
        if not selected:
            st.info("Select at least one scan to get started.")
            render_config_download(config_slot)
            st.stop()
        render_file_order(files_key, selected)
        infos = [scans[p] for p in selected]
        for info in infos:
            if info.chunks is not None:
                chunk_bytes = int(np.prod(info.chunks)) * np.dtype(info.dtype).itemsize
                if chunk_bytes > 256 * 2**20:
                    st.warning(f"{Path(info.path).name}: each Zarr chunk expands to "
                               f"{chunk_bytes / 2**30:.2f} GiB. Loading requires additional "
                               "memory for decompression and may be slow.")
        with st.expander("Curve labels", expanded=False):
            labels = [st.text_input(Path(p).name, value=suggested_label(p), key=f"label:{p}").strip()
                      for p in selected]
        if any(not label for label in labels) or len(set(labels)) != len(labels):
            st.warning("Give each selected file a nonempty, unique curve label.")
            render_config_download(config_slot)
            st.stop()

    if view == "2D scan map":
        render_scan_map(infos, labels)
    else:
        render_spectra(infos, labels, scans, config_slot)
    render_config_download(config_slot)
