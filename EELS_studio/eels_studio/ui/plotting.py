"""Shared spectrum styles, axis labels, display rules, and figure downloads."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Plotly inspects sys.modules without waiting for concurrent optional imports.
# Complete pandas initialization before constructing a Plotly trace.
import pandas  # noqa: F401
import plotly.graph_objects as go
import streamlit as st

from eels_studio.core.display import spectrum_display_intensity
from eels_studio.core.background import ANALYTIC_MODELS

COLORS = ["#137c78", "#dd7848", "#626cc6", "#c24c79", "#7a9845", "#428fbd"]
LINE_STYLES = {
    "Solid": ("solid", "-"),
    "Dashed": ("dash", "--"),
    "Dotted": ("dot", ":"),
    "Dash-dot": ("dashdot", "-."),
}


def intensity_label(mode, normalize, corrected=False, weighted=False):
    base = "Normalized detector intensity" if normalize else "Detector intensity"
    if corrected:
        base = "Background-subtracted " + base.lower()
    if weighted or mode == "energy_squared":
        base += " × E² (meV²)"
    return f"log10({base.lower()})" if mode == "log10" else base


@st.cache_data(show_spinner=False, max_entries=8)
def figure_downloads(curves, mode, x_limits, normalize, dpi=300, corrected=False, weighted=False):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for curve in curves:
        style = curve["style"]
        ax.plot(curve["energy"], spectrum_display_intensity(curve, mode, corrected, weighted),
                label=curve["label"], color=style["color"], linewidth=style["width"],
                linestyle=LINE_STYLES[style["line_style"]][1])
    ax.set_xlabel("Energy loss (meV)")
    ax.set_ylabel(intensity_label(mode, normalize, corrected, weighted))
    ax.set_xlim(*x_limits)
    ax.grid(alpha=0.16)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    result = {}
    for extension in ("svg", "png"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=dpi, bbox_inches="tight")
        result[extension] = buffer.getvalue()
    plt.close(fig)
    return result


def config_caption(config):
    stage = "intensity × E²" if config.intensity_mode == "energy_squared" else "linear intensity"
    if config.method in ANALYTIC_MODELS:
        segments = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in config.segments)
        return (f"{config.method} · segments [{segments}] meV (x/{config.energy_factor:g}) · "
                f"{stage} after optional Gaussian broadening")
    parameters = (f"log10(λ)={config.log10_lambda:g}, tol={config.tolerance:g}, max iterations={config.max_iterations}"
                  if config.method == "arPLS" else f"half-window={config.half_window_mev:g} meV")
    domain = "each full recorded domain" if config.domain == "full" else f"requested {config.energy_min:g}–{config.energy_max:g} meV"
    return f"{config.method} · {parameters} · {domain} · {stage} after optional Gaussian broadening"
