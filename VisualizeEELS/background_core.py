"""Pure 1D background fitting. Inputs are linear spectra after broadening."""
from dataclasses import dataclass, field
import hashlib
import json
import warnings
from typing import Callable

import numpy as np
import pybaselines
from pybaselines import Baseline

from eels_core import curve_identity_key

MIN_FIT_BINS = 8
PROCESSING_ORDER = ["detector integration / optional full-probe normalization", "energy ordering",
                    "optional Gaussian broadening", "background estimation and subtraction",
                    "display transform"]


@dataclass(frozen=True)
class BackgroundConfig:
    method: str = "arPLS"
    domain: str = "selected"
    energy_min: float | None = None
    energy_max: float | None = None
    log10_lambda: float = 5.0
    tolerance: float = 1e-3
    max_iterations: int = 100
    half_window_mev: float = 10.0


@dataclass(frozen=True)
class FitDiagnostics:
    requested_bounds: tuple[float, float]
    actual_bounds: tuple[float, float]
    fitted_bins: int
    spacing_mev: float
    status: str
    valid: bool
    contains_zero: bool
    tol_history: tuple[float, ...] = ()
    warnings: tuple[str, ...] = ()
    half_window_bins: int | None = None
    effective_half_window_mev: float | None = None
    package_version: str = pybaselines.__version__


@dataclass(frozen=True)
class BackgroundResult:
    energy: np.ndarray
    input: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    validity_mask: np.ndarray
    diagnostics: FitDiagnostics


def fit_background(energy, intensity, config: BackgroundConfig) -> BackgroundResult:
    """Fit one contiguous domain, never extrapolate or change the input grid.

    Selected limits must be covered in full. End bins are selected inward
    (energy >= minimum, energy <= maximum); at least eight bins are required.
    SNIP converts meV to the nearest integer bin, with half ties rounded up.
    """
    if np.iscomplexobj(energy) or np.iscomplexobj(intensity):
        raise ValueError("Energy and intensity must be real")
    x, y = np.array(energy, dtype=float, copy=True), np.array(intensity, dtype=float, copy=True)
    if x.ndim != 1 or y.shape != x.shape or x.size < MIN_FIT_BINS:
        raise ValueError("Require matching 1D arrays with at least eight fitted bins")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Energy and intensity must be finite")
    steps = np.diff(x)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-8, atol=0):
        raise ValueError("Require a strictly increasing, uniform energy axis")
    if config.method not in ("arPLS", "SNIP") or config.domain not in ("selected", "full"):
        raise ValueError("Unknown background method or fit domain")
    if not np.isfinite(config.log10_lambda) or not 2 <= config.log10_lambda <= 10:
        raise ValueError("log10(lambda) must be between 2 and 10")
    if not np.isfinite(config.tolerance) or not 0 < config.tolerance < 1:
        raise ValueError("Tolerance must be finite and between zero and one")
    if isinstance(config.max_iterations, bool) or not isinstance(config.max_iterations, (int, np.integer)) or config.max_iterations < 1:
        raise ValueError("Maximum iterations must be a positive integer")
    if not np.isfinite(config.half_window_mev) or config.half_window_mev <= 0:
        raise ValueError("SNIP half-window must be finite and positive")
    if config.domain == "selected":
        bounds = (config.energy_min, config.energy_max)
        if any(v is None for v in bounds) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Fit bounds must be finite and strictly ordered")
        if bounds[0] < x[0] or bounds[1] > x[-1]:
            raise ValueError(f"Requested interval {bounds} is not covered by [{x[0]:g}, {x[-1]:g}] meV")
    else:
        bounds = (float(x[0]), float(x[-1]))
    mask = (x >= bounds[0]) & (x <= bounds[1])
    if np.count_nonzero(mask) < MIN_FIT_BINS:
        raise ValueError("The fitting interval must contain at least eight fitted bins")
    fitted_x, fitted_y = x[mask], y[mask]
    half_bins = None
    if config.method == "SNIP":
        ratio = config.half_window_mev / steps[0]
        if not np.isfinite(ratio) or ratio > len(fitted_y):
            raise ValueError("SNIP half-window exceeds the fitting interval")
        half_bins = int(np.floor(ratio + 0.5))
        if not 1 <= half_bins <= (len(fitted_y) - 1) // 2:
            raise ValueError(f"SNIP half-window converts to {half_bins} bins; require 1–{(len(fitted_y) - 1) // 2}. No clamping is performed.")
    history = ()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if config.method == "arPLS" and np.all(fitted_y == fitted_y[0]):
            baseline = np.full_like(fitted_y, fitted_y[0])
            status, valid = "exact constant input", True
        elif config.method == "arPLS":
            baseline, params = Baseline(x_data=fitted_x, check_finite=True).arpls(
                fitted_y, lam=10.0 ** config.log10_lambda, diff_order=2,
                tol=config.tolerance, max_iter=config.max_iterations)
            history = tuple(float(v) for v in params["tol_history"])
            valid = bool(history and np.isfinite(history).all() and history[-1] <= config.tolerance)
            status = "converged" if valid else "not converged"
        else:
            baseline, _ = Baseline(x_data=fitted_x, check_finite=True).snip(
                fitted_y, max_half_window=half_bins, decreasing=True, filter_order=2,
                smooth_half_window=None)
            status, valid = "completed", True
        residual = fitted_y - baseline
    messages = tuple(str(w.message) for w in caught)
    if messages:
        status, valid = "library warning: " + "; ".join(messages), False
    if np.shape(baseline) != fitted_y.shape or not np.isfinite(baseline).all() or not np.isfinite(residual).all():
        status, valid = "numerically invalid output", False
    full_baseline, corrected = np.full_like(y, np.nan), np.full_like(y, np.nan)
    if valid:
        full_baseline[mask], corrected[mask] = baseline, residual
    else:
        mask[:] = False
    diagnostics = FitDiagnostics(tuple(float(v) for v in bounds),
        (float(fitted_x[0]), float(fitted_x[-1])), len(fitted_y), float(steps[0]), status, valid,
        bool(bounds[0] <= 0 <= bounds[1]), history, messages, half_bins,
        float(half_bins * steps[0]) if half_bins is not None else None)
    for array in (x, y, full_baseline, corrected, mask):
        array.setflags(write=False)
    return BackgroundResult(x, y, full_baseline, corrected, mask, diagnostics)


def corrected_display(values, mode):
    """Keep signed residuals in linear mode; leave gaps for nonpositive log data."""
    values = np.asarray(values, dtype=float)
    if mode != "log10":
        return values.copy()
    output = np.full_like(values, np.nan)
    positive = np.isfinite(values) & (values > 0)
    output[positive] = np.log10(values[positive])
    return output


def initial_bounds(curves):
    """Prefer common positive coverage with eight bins in every curve."""
    low = max(float(c["energy"][0]) for c in curves)
    high = min(float(c["energy"][-1]) for c in curves)
    positives = [c["energy"][c["energy"] > max(0, low)] for c in curves]
    if all(len(p) for p in positives):
        positive_low = max(float(p[0]) for p in positives)
        if all(np.count_nonzero((c["energy"] >= positive_low) & (c["energy"] <= high)) >= MIN_FIT_BINS for c in curves):
            return positive_low, high
    return low, high  # If no valid common interval exists, explicit fitting validation explains it.


def input_fingerprints(curves, settings, revisions):
    """Labels, styles and view limits are deliberately excluded."""
    processing = {k: v for k, v in settings.items() if k not in ("plot", "probe_positions_xy")}
    result = {}
    for curve in curves:
        identity = curve_identity_key(curve)
        digest = hashlib.sha256(json.dumps([identity, processing, revisions[curve["path"]]], sort_keys=True).encode())
        for name in ("energy", "intensity"):
            digest.update(np.asarray(curve[name], dtype="<f8").tobytes())
        result[identity] = digest.hexdigest()
    return result


@dataclass
class BackgroundState:
    draft: BackgroundConfig | None = None
    preview: BackgroundResult | None = None
    preview_key: tuple | None = None
    applied_config: BackgroundConfig | None = None
    applied_results: dict[str, BackgroundResult] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    source_revisions: dict[str, tuple[int, int]] = field(default_factory=dict)
    signal: str = "Input"

    def sync_inputs(self, fingerprints):
        changed = self.fingerprints != fingerprints
        if changed:
            self.reset()
            self.fingerprints = fingerprints.copy()
        return changed

    def reset(self):
        self.preview = None
        self.preview_key = None
        self.applied_config = None
        self.applied_results = {}
        self.signal = "Input"

    def apply(self, curves, config, fitter: Callable = fit_background):
        results, failures = {}, {}
        for curve in curves:
            identity = curve_identity_key(curve)
            try:
                result = fitter(curve["energy"], curve["intensity"], config)
                if not result.diagnostics.valid:
                    failures[identity] = result.diagnostics.status
                else:
                    results[identity] = result
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                failures[identity] = str(exc)
        if not failures:
            self.applied_config, self.applied_results = config, results
            self.signal = "Corrected"
        return failures


def config_caption(config):
    parameters = (f"log10(λ)={config.log10_lambda:g}, tol={config.tolerance:g}, max iterations={config.max_iterations}"
                  if config.method == "arPLS" else f"half-window={config.half_window_mev:g} meV")
    domain = "each full recorded domain" if config.domain == "full" else f"requested {config.energy_min:g}–{config.energy_max:g} meV"
    return f"{config.method} · {parameters} · {domain} · fitted after optional Gaussian broadening"
