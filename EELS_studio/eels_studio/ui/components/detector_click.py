"""Relay native Plotly heatmap clicks to a Streamlit callback without remote assets."""
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

from eels_studio.core.spectra import detector_offsets_from_click

# Read once at import time rather than on every Streamlit rerun.
_DETECTOR_CLICK_JS = Path(__file__).with_name("detector_click.js").read_text()


def register_detector_click_bridge():
    # Register in the active app runtime (also supports separate AppTest runs).
    return components.component("eels_detector_click", js=_DETECTOR_CLICK_JS)


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
