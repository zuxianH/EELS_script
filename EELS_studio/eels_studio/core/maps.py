"""Energy selection and energy-axis broadening of detector scan maps."""
import re

import numpy as np

from eels_studio.core.spectra import gaussian_broaden_spectrum


_NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_RANGE_PATTERN = re.compile(rf"^(?P<lo>{_NUMBER})-(?P<hi>{_NUMBER})$")


def parse_map_energies(text):
    """Parse map energy requests separated by commas, semicolons, or whitespace.

    A plain value like "20" requests the single nearest stored bin. A range like
    "10-20" requests every bin whose energy falls within that window, inclusive.
    Returns a list of (lo, hi) tuples, with lo == hi for plain values.
    """
    tokens = [token for token in re.split(r"[,;\s]+", text.strip()) if token]
    if not tokens:
        raise ValueError("Enter at least one map energy or range, for example: 20, 40, 10-20")
    requests = []
    for token in tokens:
        match = _RANGE_PATTERN.match(token)
        if match:
            lo, hi = float(match["lo"]), float(match["hi"])
        else:
            try:
                lo = hi = float(token)
            except ValueError:
                raise ValueError(
                    "Enter numeric map energies or ranges separated by commas, for example: 20, 40, 10-20"
                ) from None
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("Map energies must be finite numbers")
        if lo > hi:
            raise ValueError(f"Range '{token}' must have its lower bound first, for example: 10-20")
        requests.append((lo, hi))
    return requests


def selected_map_bins(axis, requests, ordering):
    """Group requests mapping to the same set of stored bins, preserving input order.

    A point request (lo == hi) picks the single nearest bin. A range request sums
    every bin whose energy falls within [lo, hi], inclusive.
    """
    raw_indices = np.arange(len(axis))
    if ordering == "Unshifted FFT":
        raw_indices = np.fft.fftshift(raw_indices)
    bins = {}
    order = []
    for lo, hi in requests:
        if lo == hi:
            axis_indices = (int(np.argmin(np.abs(axis - lo))),)
        else:
            axis_indices = tuple(i for i in range(len(axis)) if lo <= axis[i] <= hi)
            if not axis_indices:
                raise ValueError(f"No energy bins fall within {lo:g}-{hi:g} meV")
        if axis_indices not in bins:
            bins[axis_indices] = dict(
                axis_indices=list(axis_indices),
                energy_indices=[int(raw_indices[i]) for i in axis_indices],
                bin_energies_mev=[float(axis[i]) for i in axis_indices],
                requested=[])
            order.append(axis_indices)
        bins[axis_indices]["requested"].append([lo, hi])
    return [bins[key] for key in order]


def map_broadening_weights(axis, axis_index, sigma_mev):
    """Gaussian kernel weights, by axis position, for broadening one map across nearby energy bins.

    Reuses gaussian_broaden_spectrum on a one-hot vector so the reflect-boundary
    and truncation behavior exactly matches spectrum broadening.
    """
    if sigma_mev <= 0:
        return {axis_index: 1.0}
    onehot = np.zeros(len(axis))
    onehot[axis_index] = 1.0
    weights = gaussian_broaden_spectrum(axis, onehot, sigma_mev)
    return {int(i): float(weights[i]) for i in np.flatnonzero(weights)}


def broadened_scan_map(entry, axis, ordering, sigma_mev, read_map):
    """Reduce selected bins in the original order, retaining only reduced maps.

    read_map(raw_index) supplies one detector-integrated probe map. The interface
    can cache that read without introducing caching or session state here.
    """
    raw_indices = np.arange(len(axis))
    if ordering == "Unshifted FFT":
        raw_indices = np.fft.fftshift(raw_indices)
    total = None
    for axis_index in entry["axis_indices"]:
        for index, weight in map_broadening_weights(axis, axis_index, sigma_mev).items():
            contribution = weight * read_map(int(raw_indices[index]))
            total = contribution if total is None else total + contribution
    return total
