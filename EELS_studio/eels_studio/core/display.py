"""Shared numerical display transforms for interactive plots and figure exports."""
import numpy as np


def display_intensity(values, mode):
    if mode == "log10":
        return np.log10(np.clip(values, np.finfo(float).tiny, None))
    return values


def corrected_display(values, mode):
    """Keep signed residuals in linear mode; leave gaps for nonpositive log data."""
    values = np.asarray(values, dtype=float)
    if mode != "log10":
        return values.copy()
    output = np.full_like(values, np.nan)
    positive = np.isfinite(values) & (values > 0)
    output[positive] = np.log10(values[positive])
    return output


def spectrum_display_intensity(curve, mode, corrected=False, weighted=False):
    """Apply display scaling without multiplying an already weighted fit twice."""
    if mode == "energy_squared":
        return curve["intensity"] if weighted else curve["intensity"] * np.square(curve["energy"])
    transform = corrected_display if corrected else display_intensity
    return transform(curve["intensity"], mode)


def map_display(intensity, logarithmic):
    if not logarithmic:
        return intensity
    # Mask nonpositive bins so zeros do not stretch the log color range to -308.
    result = np.full(intensity.shape, np.nan)
    positive = intensity > 0
    result[positive] = np.log10(intensity[positive])
    return result
