"""Relay native Plotly heatmap clicks to a Streamlit callback without remote assets."""
from pathlib import Path

import streamlit.components.v2 as components

def register_detector_click_bridge():
    # Register in the active app runtime (also supports separate AppTest runs).
    return components.component(
        "eels_detector_click",
        js=Path(__file__).with_name("detector_click.js").read_text(),
    )
