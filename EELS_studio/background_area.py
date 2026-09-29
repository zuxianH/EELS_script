"""Signed area under a spectrum over an exact energy interval."""
import numpy as np


def integrate_area(energy, intensity, minimum, maximum):
    """Trapezoidal integration with linear interpolation at both endpoints.

    Reject uncovered intervals and gaps instead of extrapolating or connecting
    masked values. Intensities must be the original linear values, not log10.
    """
    x, y = np.asarray(energy, dtype=float), np.asarray(intensity, dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 2
            or not np.isfinite(x).all() or not np.all(np.diff(x) > 0)):
        raise ValueError('Area requires an increasing energy grid with at least two samples.')
    if not np.isfinite([minimum, maximum]).all() or minimum >= maximum:
        raise ValueError('Area minimum must be less than maximum.')
    if minimum < x[0] or maximum > x[-1]:
        raise ValueError('Choose an area range within the recorded spectrum.')
    left = np.searchsorted(x, minimum, side='right') - 1
    right = np.searchsorted(x, maximum, side='left')
    support_x, support_y = x[left:right + 1], y[left:right + 1]
    if not np.isfinite(support_y).all():
        raise ValueError('Choose an area range entirely within valid samples of the corrected preview.')
    inner = (support_x > minimum) & (support_x < maximum)
    selected_x = np.concatenate(([minimum], support_x[inner], [maximum]))
    endpoints = np.interp([minimum, maximum], support_x, support_y)
    selected_y = np.concatenate(([endpoints[0]], support_y[inner], [endpoints[1]]))
    return float(np.trapezoid(selected_y, selected_x))


def symmetric_bounds(center, half_width):
    """Return center ± half-width without clipping either side."""
    if not np.isfinite([center, half_width]).all() or half_width <= 0:
        raise ValueError('Choose a finite center and a positive half-width.')
    return float(center - half_width), float(center + half_width)


def normalize_to_original(selected_corrected_area, energy, original_intensity):
    """Return the whole original area and selected corrected / original area.

    The denominator covers the complete recorded energy range, independently
    of background fitting and the plot viewport. A cancelling or zero total
    cannot provide a meaningful normalized value.
    """
    x, y = np.asarray(energy, dtype=float), np.asarray(original_intensity, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError('Normalization requires at least two recorded energy samples.')
    total = integrate_area(x, y, x[0], x[-1])
    magnitude = integrate_area(x, np.abs(y), x[0], x[-1])
    fraction = (None if abs(total) <= max(1e-12 * magnitude, np.finfo(float).tiny)
                else float(selected_corrected_area / total))
    return total, fraction
