"""Relay box-selection events from the local Plotly rectangle component."""
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

from eels_studio.core.spectra import rectangle_from_plot

_RECTANGLE_SELECT_JS = Path(__file__).with_name("rectangle_select.js").read_text()


def register_rectangle_select():
    return components.component("eels_angle_rectangle", js=_RECTANGLE_SELECT_JS)


def receive_rectangle():
    event = st.session_state.get("angle_rectangle", {}).get("selected")
    context = st.session_state.get("angle_rectangle_context")
    if not event or not context or event.get("preview_id") != context["preview_id"]:
        return
    try:
        bounds = rectangle_from_plot(context["shape"], event["x0"], event["x1"],
                                     event["y0"], event["y1"])
        for key, value in zip(context["bound_keys"], bounds):
            st.session_state[key] = value
    except (KeyError, TypeError, ValueError) as exc:
        st.session_state["angle_rectangle_error"] = str(exc)
