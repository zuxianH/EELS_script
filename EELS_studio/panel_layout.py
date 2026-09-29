"""Drag-resizable spectrum workspace with layout stored in session settings."""
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

# Keep the implementation, but avoid its browser resize observers for now.
ADJUSTABLE_PANELS_ENABLED = False

DEFAULTS = {
    "panel_detector_width": 1 / 3,
    "panel_spectrum_height": 520,
    "panel_detector_height": 420,
    "panel_detector_first": True,
}


def register_panel_layout():
    return components.component("eels_panel_layout", js=Path(__file__).with_suffix(".js").read_text())


def layout_settings():
    if not ADJUSTABLE_PANELS_ENABLED:
        return DEFAULTS.copy()
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    return {key: st.session_state[key] for key in DEFAULTS}


def update_panel_layout():
    event = st.session_state.get("panel_layout_bridge", {}).get("layout")
    if not isinstance(event, dict):
        return
    from workspace_config import _setting
    try:
        values = {key: _setting(key, event[key]) for key in DEFAULTS}
    except (KeyError, ValueError, TypeError, OverflowError):
        return
    st.session_state.update(values)


def reset_panel_layout():
    st.session_state.update(DEFAULTS)
