#!/usr/bin/env python3
"""Training-free, joint-lambda arPLS subtraction for EELS spectra.

Automatic lambda selection uses the maximum curvature of a normalized joint
L-curve.  It is a reproducible model-selection heuristic, not knowledge of the
true physical background, and it cannot establish that a fitted background is
physically unique.

No momentum selection, detector masking, interpolation, or machine learning is
performed here.  Every input spectrum must already be momentum-integrated and
sampled on the same energy grid.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve


# ---------------------------------------------------------------------------
# User-adjustable parameters
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# Choose exactly one input mode.  The existing stacked TACAW file is selected
# by default so this script runs immediately in this project directory.
#
# 1. Separate one-dimensional energy/intensity NPY files.  Paths relative to
#    this script are accepted.  Add as many entries as needed.
DATASETS: list[dict[str, str]] = [
    # {
    #     "label": "Spectrum 1",
    #     "energy_file": "energy_1.npy",
    #     "intensity_file": "intensity_1.npy",
    # },
    # {
    #     "label": "Spectrum 2",
    #     "energy_file": "energy_2.npy",
    #     "intensity_file": "intensity_2.npy",
    # },
]

# 2. One NPZ file with keys "energy" and "intensity".  The intensity array
#    may have shape (n_energy,) or (n_spectra, n_energy).
NPZ_INPUT_FILE: str | None = None  # For example: "spectra.npz"

# 3. One stacked NPY array with shape (n_spectra, n_energy, 2), where column 0
#    is the energy axis and column 1 is intensity.  Set this to None when using
#    DATASETS or NPZ_INPUT_FILE above.
STACKED_NPY_INPUT_FILE: str | None = (
    "SiC_eels_thicknessdata_center_40px_new.npy"
)

ENERGY_MIN = 5.0
ENERGY_MAX = 200.0  # meV

ENERGY_GRID_RTOL = 1e-7
ENERGY_GRID_ATOL = 1e-9

# Smaller lambda -> a more flexible baseline.
# Larger lambda  -> a smoother baseline.
LAMBDA_GRID = np.logspace(2, 10, 33)
MANUAL_LAMBDA: float | None = None

ARPLS_RATIO = 1e-6
ARPLS_MAX_ITER = 100

DIAGNOSTIC_SPECTRUM_INDEX = 0
L_CURVE_EPSILON = 1e-15
MAX_NONCONVERGED_FRACTION = 0.20
MIN_MEANINGFUL_CURVATURE = 1e-6
L_CURVE_POINT_TOLERANCE = 1e-10
NEIGHBOR_BASELINE_CHANGE_THRESHOLD = 0.10

OUTPUT_FILE = SCRIPT_DIRECTORY / "joint_arpls_results.npz"
CSV_OUTPUT_FILE = SCRIPT_DIRECTORY / "joint_arpls_lambda_diagnostics.csv"
SPECTRUM_TEXT_DIRECTORY = SCRIPT_DIRECTORY / "joint_arpls_spectra"
PLOT_DIRECTORY = SCRIPT_DIRECTORY / "joint_arpls_plots"
PLOT_DPI = 250
SHOW_PLOTS = True


def _resolve_path(filename: str | Path, base_directory: str | Path) -> Path:
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = Path(base_directory) / path
    return path


def _sort_and_validate_spectrum(
    energy: np.ndarray,
    intensity: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated copies sorted in strictly increasing energy order."""
    energy = np.asarray(energy)
    intensity = np.asarray(intensity)

    if energy.ndim != 1:
        raise ValueError(f"{label}: energy must be one-dimensional; got {energy.shape}.")
    if intensity.ndim != 1:
        raise ValueError(
            f"{label}: intensity must be one-dimensional; got {intensity.shape}."
        )
    if energy.size != intensity.size:
        raise ValueError(
            f"{label}: energy has {energy.size} values but intensity has "
            f"{intensity.size}."
        )
    if energy.size < 3:
        raise ValueError(f"{label}: at least three energy points are required.")
    if not np.issubdtype(energy.dtype, np.number):
        raise TypeError(f"{label}: energy must be numeric; got {energy.dtype}.")
    if not np.issubdtype(intensity.dtype, np.number):
        raise TypeError(f"{label}: intensity must be numeric; got {intensity.dtype}.")

    energy = np.asarray(energy, dtype=float).copy()
    intensity = np.asarray(intensity, dtype=float).copy()
    if not np.all(np.isfinite(energy)):
        raise ValueError(f"{label}: energy contains non-finite values.")
    if not np.all(np.isfinite(intensity)):
        raise ValueError(f"{label}: intensity contains non-finite values.")

    order = np.argsort(energy, kind="stable")
    energy = energy[order]
    intensity = intensity[order]
    if not np.all(np.diff(energy) > 0):
        raise ValueError(
            f"{label}: energy must be strictly increasing after sorting; "
            "duplicate energy values are not allowed."
        )
    return energy, intensity


def validate_common_energy_grid(
    energy_axes: list[np.ndarray],
    rtol: float = ENERGY_GRID_RTOL,
    atol: float = ENERGY_GRID_ATOL,
) -> np.ndarray:
    """Validate equal-length, numerically matching grids without interpolation."""
    if not energy_axes:
        raise ValueError("No energy axes were supplied.")
    if rtol < 0 or atol < 0 or not np.isfinite(rtol) or not np.isfinite(atol):
        raise ValueError("Energy-grid tolerances must be finite and non-negative.")

    reference = energy_axes[0]
    for index, candidate in enumerate(energy_axes[1:], start=1):
        if candidate.shape != reference.shape or not np.allclose(
            candidate,
            reference,
            rtol=rtol,
            atol=atol,
        ):
            maximum_difference = np.nan
            if candidate.shape == reference.shape:
                maximum_difference = float(np.max(np.abs(candidate - reference)))
            raise ValueError(
                "Spectrum energy grids do not match within the configured "
                f"tolerance (first mismatch: spectrum {index}; maximum absolute "
                f"difference={maximum_difference}). A common numerical lambda is "
                "grid-dependent. Resample the spectra onto one common energy grid "
                "before running this program; interpolation is intentionally not "
                "performed here."
            )
    return reference.copy()


def load_datasets(
    datasets: list[dict[str, str]] | None = None,
    npz_file: str | Path | None = None,
    stacked_npy_file: str | Path | None = None,
    base_directory: str | Path = SCRIPT_DIRECTORY,
    grid_rtol: float = ENERGY_GRID_RTOL,
    grid_atol: float = ENERGY_GRID_ATOL,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load separate NPY pairs, one NPZ, or one stacked TACAW NPY array."""
    datasets = [] if datasets is None else datasets
    configured_modes = sum(
        (
            bool(datasets),
            npz_file is not None,
            stacked_npy_file is not None,
        )
    )
    if configured_modes == 0:
        raise ValueError(
            "No input is configured. Set exactly one of DATASETS, "
            "NPZ_INPUT_FILE, or STACKED_NPY_INPUT_FILE."
        )
    if configured_modes > 1:
        raise ValueError(
            "Multiple input modes are configured. Set exactly one of DATASETS, "
            "NPZ_INPUT_FILE, or STACKED_NPY_INPUT_FILE and disable the others."
        )

    energy_axes: list[np.ndarray] = []
    intensity_rows: list[np.ndarray] = []
    labels: list[str] = []

    if stacked_npy_file is not None:
        input_path = _resolve_path(stacked_npy_file, base_directory)
        stacked_data = np.load(input_path)
        if stacked_data.ndim != 3 or stacked_data.shape[2] != 2:
            raise ValueError(
                "Stacked NPY input must have shape (n_spectra, n_energy, 2); "
                f"got {stacked_data.shape}."
            )
        if stacked_data.shape[0] < 1:
            raise ValueError("The stacked NPY input contains no spectra.")

        for index in range(stacked_data.shape[0]):
            label = f"Spectrum {index + 1}"
            sorted_energy, sorted_intensity = _sort_and_validate_spectrum(
                stacked_data[index, :, 0],
                stacked_data[index, :, 1],
                label,
            )
            energy_axes.append(sorted_energy)
            intensity_rows.append(sorted_intensity)
            labels.append(label)
    elif npz_file is not None:
        input_path = _resolve_path(npz_file, base_directory)
        with np.load(input_path) as archive:
            missing = {"energy", "intensity"}.difference(archive.files)
            if missing:
                raise KeyError(
                    f"{input_path} is missing required NPZ key(s): "
                    f"{', '.join(sorted(missing))}."
                )
            energy_raw = np.asarray(archive["energy"])
            intensity_raw = np.asarray(archive["intensity"])

        if energy_raw.ndim != 1:
            raise ValueError(
                f"NPZ energy must have shape (n_energy,), got {energy_raw.shape}."
            )
        if intensity_raw.ndim == 1:
            intensity_raw = intensity_raw[None, :]
        elif intensity_raw.ndim != 2:
            raise ValueError(
                "NPZ intensity must have shape (n_energy,) or "
                f"(n_spectra, n_energy); got {intensity_raw.shape}."
            )
        if intensity_raw.shape[0] < 1:
            raise ValueError("The NPZ input contains no spectra.")

        for index, row in enumerate(intensity_raw):
            label = f"Spectrum {index + 1}"
            sorted_energy, sorted_intensity = _sort_and_validate_spectrum(
                energy_raw,
                row,
                label,
            )
            energy_axes.append(sorted_energy)
            intensity_rows.append(sorted_intensity)
            labels.append(label)
    else:
        for index, dataset in enumerate(datasets):
            required = {"energy_file", "intensity_file"}
            missing = required.difference(dataset)
            if missing:
                raise KeyError(
                    f"DATASETS entry {index} is missing: {', '.join(sorted(missing))}."
                )
            label = str(dataset.get("label", f"Spectrum {index + 1}"))
            energy_path = _resolve_path(dataset["energy_file"], base_directory)
            intensity_path = _resolve_path(dataset["intensity_file"], base_directory)
            energy_raw = np.load(energy_path)
            intensity_raw = np.load(intensity_path)
            sorted_energy, sorted_intensity = _sort_and_validate_spectrum(
                energy_raw,
                intensity_raw,
                label,
            )
            energy_axes.append(sorted_energy)
            intensity_rows.append(sorted_intensity)
            labels.append(label)

    energy = validate_common_energy_grid(
        energy_axes,
        rtol=grid_rtol,
        atol=grid_atol,
    )
    intensities = np.stack(intensity_rows, axis=0)

    if intensities.shape[0] == 1:
        warnings.warn(
            "Only one spectrum was supplied. Automatic lambda selection will use "
            "the same L-curve method, but the selection is not genuinely joint.",
            RuntimeWarning,
            stacklevel=2,
        )
    return energy, intensities, labels


def select_energy_range(
    energy: np.ndarray,
    intensities: np.ndarray,
    energy_min: float,
    energy_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the shared inclusive fitting interval."""
    if not np.isfinite(energy_min) or not np.isfinite(energy_max):
        raise ValueError("ENERGY_MIN and ENERGY_MAX must be finite.")
    if energy_min > energy_max:
        raise ValueError("ENERGY_MIN must not exceed ENERGY_MAX.")

    energy_mask = (energy >= energy_min) & (energy <= energy_max)
    number_selected = int(np.count_nonzero(energy_mask))
    if number_selected < 3:
        raise ValueError(
            "The selected interval must contain at least three points to build "
            f"the second-difference matrix; it contains {number_selected}."
        )
    return energy_mask, energy[energy_mask], intensities[:, energy_mask]


def build_difference_matrix(number_points: int) -> sparse.csc_matrix:
    """Construct the sparse second-order finite-difference matrix."""
    if not isinstance(number_points, (int, np.integer)) or number_points < 3:
        raise ValueError("At least three points are required for second differences.")
    diagonal_length = number_points - 2
    return sparse.diags(
        diagonals=(
            np.ones(diagonal_length),
            -2.0 * np.ones(diagonal_length),
            np.ones(diagonal_length),
        ),
        offsets=(0, 1, 2),
        shape=(diagonal_length, number_points),
        format="csc",
    )


def _solve_weighted_baseline(
    y: np.ndarray,
    weights: np.ndarray,
    smoothness_penalty: sparse.spmatrix,
    iteration: int,
) -> np.ndarray:
    weight_matrix = sparse.diags(weights, offsets=0, format="csc")
    system_matrix = weight_matrix + smoothness_penalty
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            baseline = spsolve(system_matrix, weights * y)
    except Exception as exc:
        raise RuntimeError(
            f"Sparse arPLS solve failed or was singular at iteration {iteration}."
        ) from exc
    baseline = np.asarray(baseline, dtype=float)
    if baseline.shape != y.shape or not np.all(np.isfinite(baseline)):
        raise RuntimeError(
            f"Sparse arPLS solve returned a non-finite result at iteration {iteration}."
        )
    return baseline


def arpls(
    y: np.ndarray,
    lam: float,
    ratio: float = 1e-6,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    """Estimate one arPLS baseline and return its final asymmetric weights."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"arpls expects a 1D spectrum; got shape {y.shape}.")
    if y.size < 3:
        raise ValueError("arpls requires at least three points.")
    if not np.all(np.isfinite(y)):
        raise ValueError("arpls input contains non-finite values.")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive.")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be finite and positive.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")

    difference_matrix = build_difference_matrix(y.size)
    smoothness_penalty = lam * (difference_matrix.T @ difference_matrix)
    weights = np.ones(y.size, dtype=float)
    baseline = np.zeros_like(y)
    relative_change = np.inf
    converged = False
    status = "maximum_iterations_reached"
    weights_changed_after_solve = False

    for iteration in range(1, max_iter + 1):
        baseline = _solve_weighted_baseline(
            y,
            weights,
            smoothness_penalty,
            iteration,
        )
        weights_changed_after_solve = False
        residual = y - baseline
        negative = residual[residual < 0]

        if negative.size < 2:
            status = "fewer_than_two_negative_residuals"
            break

        negative_mean = float(np.mean(negative))
        negative_std = float(np.std(negative))
        if not np.isfinite(negative_mean) or not np.isfinite(negative_std):
            status = "non_finite_negative_residual_statistics"
            break

        standard_deviation_floor = (
            np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(y))), float(np.max(np.abs(negative))))
        )
        if negative_std <= standard_deviation_floor:
            status = "near_zero_negative_residual_standard_deviation"
            break

        logistic_argument = (
            2.0 * (residual - (2.0 * negative_std - negative_mean)) / negative_std
        )
        logistic_argument = np.clip(logistic_argument, -60.0, 60.0)
        new_weights = 1.0 / (1.0 + np.exp(logistic_argument))
        if not np.all(np.isfinite(new_weights)):
            status = "non_finite_weights"
            break

        weight_norm = float(np.linalg.norm(weights))
        if not np.isfinite(weight_norm) or weight_norm <= np.finfo(float).tiny:
            status = "invalid_weight_norm"
            break
        relative_change = float(np.linalg.norm(weights - new_weights) / weight_norm)
        if not np.isfinite(relative_change):
            status = "non_finite_relative_weight_change"
            break

        weights = new_weights
        weights_changed_after_solve = True
        if relative_change < ratio:
            converged = True
            status = "converged"
            break

    # Make the returned baseline consistent with the returned final weights,
    # including at convergence and at the iteration cap.
    if weights_changed_after_solve:
        baseline = _solve_weighted_baseline(
            y,
            weights,
            smoothness_penalty,
            iteration,
        )

    info: dict[str, float | int | bool | str] = {
        "iterations": iteration,
        "converged": converged,
        "relative_weight_change": relative_change,
        "status": status,
        "lambda": float(lam),
    }
    return baseline, weights, info


def robust_intensity_scale(y: np.ndarray) -> float:
    """Calculate a scale that is robust to peaks and rejects constant spectra."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("robust_intensity_scale expects one finite 1D spectrum.")

    maximum_absolute = float(np.max(np.abs(y)))
    numerical_floor = 100.0 * np.finfo(float).eps * max(1.0, maximum_absolute)
    full_range = float(np.ptp(y))
    if full_range <= numerical_floor:
        raise ValueError(
            "The spectrum is effectively constant, so normalized joint-lambda "
            "selection is undefined."
        )

    percentile_scale = float(np.percentile(y, 95) - np.percentile(y, 5))
    if np.isfinite(percentile_scale) and percentile_scale > numerical_floor:
        return percentile_scale

    standard_deviation = float(np.std(y))
    if np.isfinite(standard_deviation) and standard_deviation > numerical_floor:
        return standard_deviation

    if np.isfinite(maximum_absolute) and maximum_absolute > numerical_floor:
        return maximum_absolute
    raise ValueError("No finite, non-zero robust intensity scale could be calculated.")


def evaluate_lambda_candidate(
    intensities: np.ndarray,
    lam: float,
    ratio: float,
    max_iter: int,
) -> dict[str, object]:
    """Evaluate normalized residual and roughness measures at one lambda."""
    intensities = np.asarray(intensities, dtype=float)
    if intensities.ndim != 2 or intensities.shape[1] < 3:
        raise ValueError(
            "intensities must have shape (n_spectra, n_energy), with at least "
            "three selected energy points."
        )
    if not np.all(np.isfinite(intensities)):
        raise ValueError("Candidate evaluation received non-finite intensities.")

    number_spectra, number_points = intensities.shape
    difference_matrix = build_difference_matrix(number_points)
    residual_measures = np.empty(number_spectra, dtype=float)
    roughness_measures = np.empty(number_spectra, dtype=float)
    baselines = np.empty_like(intensities, dtype=float)
    weights = np.empty_like(intensities, dtype=float)
    infos: list[dict[str, float | int | bool | str]] = []

    for spectrum_index, y in enumerate(intensities):
        scale = robust_intensity_scale(y)
        baseline, final_weights, info = arpls(
            y,
            lam=lam,
            ratio=ratio,
            max_iter=max_iter,
        )
        residual = y - baseline
        weight_sum = float(np.sum(final_weights))
        if not np.isfinite(weight_sum) or weight_sum <= np.finfo(float).tiny:
            raise RuntimeError(
                f"Invalid final weight sum for spectrum {spectrum_index}, "
                f"lambda={lam:.6g}."
            )

        residual_measures[spectrum_index] = np.linalg.norm(
            np.sqrt(final_weights) * residual
        ) / (scale * np.sqrt(weight_sum))
        roughness_measures[spectrum_index] = np.linalg.norm(
            difference_matrix @ baseline
        ) / (scale * np.sqrt(number_points - 2))
        baselines[spectrum_index] = baseline
        weights[spectrum_index] = final_weights
        infos.append(info)

    if not np.all(np.isfinite(residual_measures)) or not np.all(
        np.isfinite(roughness_measures)
    ):
        raise RuntimeError(f"Non-finite L-curve measure at lambda={lam:.6g}.")
    if np.any(residual_measures < 0) or np.any(roughness_measures < 0):
        raise RuntimeError(f"Negative L-curve measure at lambda={lam:.6g}.")

    return {
        "lambda": float(lam),
        "residual_measures": residual_measures,
        "roughness_measures": roughness_measures,
        "baselines": baselines,
        "weights": weights,
        "infos": infos,
        "number_converged": sum(bool(info["converged"]) for info in infos),
    }


def calculate_lcurve_curvature(
    lambda_grid: np.ndarray,
    joint_log_residual: np.ndarray,
    joint_log_roughness: np.ndarray,
) -> np.ndarray:
    """Calculate discrete parametric L-curve curvature in log10(lambda)."""
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    x = np.asarray(joint_log_residual, dtype=float)
    y = np.asarray(joint_log_roughness, dtype=float)
    if lambda_grid.ndim != 1 or x.shape != lambda_grid.shape or y.shape != x.shape:
        raise ValueError("Lambda and joint L-curve arrays must be matching 1D arrays.")
    if lambda_grid.size < 3:
        raise ValueError("At least three lambda candidates are required.")
    if np.any(~np.isfinite(lambda_grid)) or np.any(lambda_grid <= 0):
        raise ValueError("Every candidate lambda must be finite and positive.")
    if not np.all(np.diff(lambda_grid) > 0):
        raise ValueError("Candidate lambdas must be strictly increasing.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.full_like(lambda_grid, np.nan, dtype=float)

    parameter = np.log10(lambda_grid)
    edge_order = 2 if lambda_grid.size >= 3 else 1
    x_first = np.gradient(x, parameter, edge_order=edge_order)
    y_first = np.gradient(y, parameter, edge_order=edge_order)
    x_second = np.gradient(x_first, parameter, edge_order=edge_order)
    y_second = np.gradient(y_first, parameter, edge_order=edge_order)

    numerator = np.abs(x_first * y_second - y_first * x_second)
    speed_squared = x_first**2 + y_first**2
    denominator = speed_squared**1.5
    denominator_floor = np.finfo(float).eps * max(
        1.0,
        float(np.nanmax(denominator)),
    )

    curvature = np.full_like(lambda_grid, np.nan, dtype=float)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > denominator_floor)
    )
    curvature[valid] = numerator[valid] / denominator[valid]
    curvature[0] = np.nan
    curvature[-1] = np.nan
    return curvature


def _neighbor_baseline_changes(
    candidate_results: list[dict[str, object]],
    selected_index: int,
    intensities: np.ndarray,
) -> dict[str, float]:
    selected_baselines = np.asarray(
        candidate_results[selected_index]["baselines"],
        dtype=float,
    )
    scales = np.array([robust_intensity_scale(y) for y in intensities])
    changes: dict[str, float] = {}
    for name, neighbor_index in (
        ("smaller", selected_index - 1),
        ("larger", selected_index + 1),
    ):
        if 0 <= neighbor_index < len(candidate_results):
            neighbor_baselines = np.asarray(
                candidate_results[neighbor_index]["baselines"],
                dtype=float,
            )
            per_spectrum = np.sqrt(
                np.mean((neighbor_baselines - selected_baselines) ** 2, axis=1)
            ) / scales
            changes[name] = float(np.max(per_spectrum))
    return changes


def _assess_selection_reliability(
    lambda_grid: np.ndarray,
    joint_log_residual: np.ndarray,
    joint_log_roughness: np.ndarray,
    curvature: np.ndarray,
    number_converged: np.ndarray,
    number_spectra: int,
    selected_index: int,
    neighbor_changes: dict[str, float],
) -> tuple[bool, list[str]]:
    messages: list[str] = []
    total_fits = int(lambda_grid.size * number_spectra)
    nonconverged = total_fits - int(np.sum(number_converged))
    nonconverged_fraction = nonconverged / total_fits
    if nonconverged_fraction > MAX_NONCONVERGED_FRACTION:
        messages.append(
            f"{nonconverged}/{total_fits} candidate arPLS fits did not converge "
            f"({nonconverged_fraction:.1%}), exceeding the configured "
            f"{MAX_NONCONVERGED_FRACTION:.1%} threshold."
        )

    maximum_curvature = curvature[selected_index]
    if not np.isfinite(maximum_curvature):
        messages.append("The selected L-curve curvature is non-finite.")
    elif maximum_curvature <= MIN_MEANINGFUL_CURVATURE:
        messages.append(
            f"Maximum curvature ({maximum_curvature:.3e}) is nearly zero."
        )

    if selected_index in (0, lambda_grid.size - 1):
        messages.append(
            "The selected lambda is the smallest or largest candidate. Expand "
            "LAMBDA_GRID before interpreting the result."
        )
    elif selected_index in (1, lambda_grid.size - 2):
        messages.append(
            "The curvature maximum is adjacent to a lambda-grid boundary. "
            "Expand LAMBDA_GRID before interpreting the result."
        )

    coordinate_scale = max(
        1.0,
        float(np.ptp(joint_log_residual)),
        float(np.ptp(joint_log_roughness)),
    )
    point_separation = np.hypot(
        np.diff(joint_log_residual),
        np.diff(joint_log_roughness),
    )
    if np.any(point_separation <= L_CURVE_POINT_TOLERANCE * coordinate_scale):
        messages.append("The joint L-curve contains duplicate numerical points.")
    if (
        np.ptp(joint_log_residual) <= L_CURVE_POINT_TOLERANCE * coordinate_scale
        and np.ptp(joint_log_roughness)
        <= L_CURVE_POINT_TOLERANCE * coordinate_scale
    ):
        messages.append("The joint L-curve is numerically constant.")

    for direction, change in neighbor_changes.items():
        if not np.isfinite(change):
            messages.append(
                f"The normalized baseline change toward the {direction} lambda "
                "candidate is non-finite."
            )
        elif change > NEIGHBOR_BASELINE_CHANGE_THRESHOLD:
            messages.append(
                f"The baseline changes strongly toward the {direction} neighboring "
                f"lambda (maximum normalized RMS change={change:.3g}, threshold="
                f"{NEIGHBOR_BASELINE_CHANGE_THRESHOLD:.3g})."
            )
    return len(messages) == 0, messages


def select_joint_lambda(
    intensities: np.ndarray,
    lambda_grid: np.ndarray,
    ratio: float,
    max_iter: int,
) -> tuple[float, dict[str, object]]:
    """Select one common lambda using normalized, geometric-mean L-curve data."""
    intensities = np.asarray(intensities, dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    if intensities.ndim != 2 or intensities.shape[0] < 1:
        raise ValueError("intensities must be a non-empty 2D array.")
    if lambda_grid.ndim != 1 or lambda_grid.size < 3:
        raise ValueError("lambda_grid must contain at least three candidates.")
    if np.any(~np.isfinite(lambda_grid)) or np.any(lambda_grid <= 0):
        raise ValueError("Every lambda candidate must be finite and positive.")
    if not np.all(np.diff(lambda_grid) > 0):
        raise ValueError("lambda_grid must be strictly increasing.")

    candidate_results: list[dict[str, object]] = []
    joint_log_residual = np.empty(lambda_grid.size, dtype=float)
    joint_log_roughness = np.empty(lambda_grid.size, dtype=float)
    number_converged = np.empty(lambda_grid.size, dtype=int)

    for candidate_index, lam in enumerate(lambda_grid):
        result = evaluate_lambda_candidate(
            intensities,
            lam=float(lam),
            ratio=ratio,
            max_iter=max_iter,
        )
        candidate_results.append(result)
        residual_measures = np.asarray(result["residual_measures"], dtype=float)
        roughness_measures = np.asarray(result["roughness_measures"], dtype=float)
        joint_log_residual[candidate_index] = float(
            np.mean(np.log(residual_measures + L_CURVE_EPSILON))
        )
        joint_log_roughness[candidate_index] = float(
            np.mean(np.log(roughness_measures + L_CURVE_EPSILON))
        )
        number_converged[candidate_index] = int(result["number_converged"])

    curvature = calculate_lcurve_curvature(
        lambda_grid,
        joint_log_residual,
        joint_log_roughness,
    )
    valid_interior = np.flatnonzero(np.isfinite(curvature))
    preselection_warnings: list[str] = []
    if valid_interior.size:
        selected_index = int(valid_interior[np.argmax(curvature[valid_interior])])
    else:
        selected_index = lambda_grid.size // 2
        preselection_warnings.append(
            "No finite interior L-curve curvature was available; the central "
            "lambda was used as an explicit, unreliable fallback."
        )

    selected_lambda = float(lambda_grid[selected_index])
    neighbor_changes = _neighbor_baseline_changes(
        candidate_results,
        selected_index,
        intensities,
    )
    reliable, reliability_warnings = _assess_selection_reliability(
        lambda_grid=lambda_grid,
        joint_log_residual=joint_log_residual,
        joint_log_roughness=joint_log_roughness,
        curvature=curvature,
        number_converged=number_converged,
        number_spectra=intensities.shape[0],
        selected_index=selected_index,
        neighbor_changes=neighbor_changes,
    )
    selection_warnings = preselection_warnings + reliability_warnings
    reliable = reliable and not preselection_warnings
    maximum_curvature = float(curvature[selected_index])

    selection_info = {
        "selected_lambda": selected_lambda,
        "selected_index": selected_index,
        "maximum_curvature": maximum_curvature,
        "selection_reliable": reliable,
        "warnings": selection_warnings,
        "neighbor_baseline_changes": neighbor_changes,
    }
    diagnostics: dict[str, object] = {
        "lambda_grid": lambda_grid.copy(),
        "joint_log_residual": joint_log_residual,
        "joint_log_roughness": joint_log_roughness,
        "curvature": curvature,
        "number_converged": number_converged,
        "selection_info": selection_info,
        "candidate_results": candidate_results,
    }
    return selected_lambda, diagnostics


def apply_common_lambda(
    intensities: np.ndarray,
    selected_lambda: float,
    ratio: float,
    max_iter: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, float | int | bool | str]],
]:
    """Rerun every selected spectrum using exactly one common lambda."""
    intensities = np.asarray(intensities, dtype=float)
    if intensities.ndim != 2:
        raise ValueError("intensities must have shape (n_spectra, n_energy).")

    baselines = np.empty_like(intensities, dtype=float)
    final_weights = np.empty_like(intensities, dtype=float)
    infos: list[dict[str, float | int | bool | str]] = []
    for index, y in enumerate(intensities):
        baseline, weights, info = arpls(
            y,
            lam=selected_lambda,
            ratio=ratio,
            max_iter=max_iter,
        )
        baselines[index] = baseline
        final_weights[index] = weights
        infos.append(info)
    corrected = intensities - baselines
    return baselines, corrected, final_weights, infos


def plot_joint_diagnostics(
    diagnostics: dict[str, object],
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the joint L-curve and curvature used for automatic selection."""
    lambda_grid = np.asarray(diagnostics["lambda_grid"], dtype=float)
    log_residual = np.asarray(diagnostics["joint_log_residual"], dtype=float)
    log_roughness = np.asarray(diagnostics["joint_log_roughness"], dtype=float)
    curvature = np.asarray(diagnostics["curvature"], dtype=float)
    selection_info = dict(diagnostics["selection_info"])
    selected_index = int(selection_info["selected_index"])
    parameter = np.log10(lambda_grid)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    points = axes[0].scatter(
        log_residual,
        log_roughness,
        c=parameter,
        cmap="viridis",
        s=42,
        zorder=3,
    )
    axes[0].plot(log_residual, log_roughness, color="0.55", linewidth=1.0)
    axes[0].scatter(
        log_residual[selected_index],
        log_roughness[selected_index],
        marker="*",
        s=240,
        facecolor="none",
        edgecolor="red",
        linewidth=1.8,
        label=fr"Selected $\lambda={lambda_grid[selected_index]:.2e}$",
        zorder=4,
    )
    axes[0].set_xlabel(r"$\log R_{\mathrm{joint}}$")
    axes[0].set_ylabel(r"$\log Q_{\mathrm{joint}}$")
    axes[0].set_title("Normalized joint L-curve")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    colorbar = figure.colorbar(points, ax=axes[0])
    colorbar.set_label(r"$\log_{10}\lambda$")

    axes[1].plot(parameter, curvature, marker="o", markersize=3.5)
    axes[1].axvline(
        parameter[selected_index],
        color="red",
        linestyle="--",
        label=fr"Selected $\lambda={lambda_grid[selected_index]:.2e}$",
    )
    axes[1].set_xlabel(r"$\log_{10}\lambda$")
    axes[1].set_ylabel(r"L-curve curvature $\kappa$")
    axes[1].set_title("Discrete L-curve curvature")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    return figure, axes


def plot_spectrum_results(
    energy: np.ndarray,
    intensities: np.ndarray,
    baselines: np.ndarray,
    corrected: np.ndarray,
    selected_lambda: float,
    labels: list[str],
) -> list[plt.Figure]:
    """Create one original/background/corrected figure per spectrum."""
    figures: list[plt.Figure] = []
    for index, (intensity, baseline, corrected_spectrum) in enumerate(
        zip(intensities, baselines, corrected)
    ):
        figure, axes = plt.subplots(figsize=(8.5, 5.2))
        axes.plot(energy, intensity, label="Original")
        axes.plot(
            energy,
            baseline,
            label=fr"arPLS background, $\lambda={selected_lambda:.2e}$",
        )
        axes.plot(energy, corrected_spectrum, label="Background-subtracted")
        axes.set_xlabel("Energy loss (meV)")
        axes.set_ylabel("Intensity")
        axes.set_title(f"{labels[index]}: common-lambda arPLS subtraction")
        axes.legend()
        axes.grid(alpha=0.2)
        figure.tight_layout()
        figures.append(figure)
    return figures


def _candidate_indices_for_plot(
    lambda_grid: np.ndarray,
    selected_lambda: float,
) -> list[tuple[str, float]]:
    smaller = np.flatnonzero(lambda_grid < selected_lambda)
    larger = np.flatnonzero(lambda_grid > selected_lambda)
    candidates: list[tuple[str, float]] = []
    if smaller.size:
        candidates.append(("Immediately smaller", float(lambda_grid[smaller[-1]])))
    candidates.append(("Selected", float(selected_lambda)))
    if larger.size:
        candidates.append(("Immediately larger", float(lambda_grid[larger[0]])))
    return candidates


def plot_candidate_baselines(
    energy: np.ndarray,
    intensity: np.ndarray,
    lambda_grid: np.ndarray,
    selected_lambda: float,
    ratio: float,
    max_iter: int,
    label: str,
) -> tuple[plt.Figure, list[dict[str, float | int | bool | str]]]:
    """Overlay the selected baseline and its nearest grid neighbors."""
    figure, axes = plt.subplots(figsize=(9, 5.5))
    axes.plot(energy, intensity, color="black", linewidth=1.5, label="Original")
    infos: list[dict[str, float | int | bool | str]] = []
    for candidate_label, lam in _candidate_indices_for_plot(
        np.asarray(lambda_grid, dtype=float),
        selected_lambda,
    ):
        baseline, _, info = arpls(
            intensity,
            lam=lam,
            ratio=ratio,
            max_iter=max_iter,
        )
        line_width = 2.2 if candidate_label == "Selected" else 1.2
        axes.plot(
            energy,
            baseline,
            linewidth=line_width,
            label=fr"{candidate_label}: $\lambda={lam:.2e}$",
        )
        infos.append(info)
    axes.set_xlabel("Energy loss (meV)")
    axes.set_ylabel("Intensity")
    axes.set_title(f"{label}: neighboring-lambda baseline stability")
    axes.legend()
    axes.grid(alpha=0.2)
    figure.tight_layout()
    return figure, infos


def _manual_diagnostics(
    lambda_grid: np.ndarray,
    selected_lambda: float,
) -> dict[str, object]:
    """Create explicit not-evaluated diagnostics for a manual override."""
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    selected_index = int(np.argmin(np.abs(np.log(lambda_grid / selected_lambda))))
    selection_info = {
        "selected_lambda": float(selected_lambda),
        "selected_index": selected_index,
        "maximum_curvature": np.nan,
        "selection_reliable": False,
        "warnings": [
            "MANUAL_LAMBDA bypassed automatic joint selection; L-curve "
            "reliability was not evaluated."
        ],
        "neighbor_baseline_changes": {},
    }
    return {
        "lambda_grid": lambda_grid.copy(),
        "joint_log_residual": np.full(lambda_grid.shape, np.nan),
        "joint_log_roughness": np.full(lambda_grid.shape, np.nan),
        "curvature": np.full(lambda_grid.shape, np.nan),
        "number_converged": np.zeros(lambda_grid.shape, dtype=int),
        "selection_info": selection_info,
        "candidate_results": [],
    }


def save_results(
    output_file: str | Path,
    csv_output_file: str | Path,
    spectrum_text_directory: str | Path,
    energy: np.ndarray,
    intensities: np.ndarray,
    energy_mask: np.ndarray,
    baseline_selected: np.ndarray,
    corrected_selected: np.ndarray,
    weights_selected: np.ndarray,
    selected_lambda: float,
    diagnostics: dict[str, object],
    labels: list[str],
    energy_min: float,
    energy_max: float,
    ratio: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Save full-grid NPZ, candidate CSV, and one full-grid text file per spectrum.

    Background, corrected intensity, and final weights are NaN outside the
    fitted energy interval.  Original intensities remain present everywhere.
    """
    background = np.full_like(intensities, np.nan, dtype=float)
    corrected = np.full_like(intensities, np.nan, dtype=float)
    final_weights = np.full_like(intensities, np.nan, dtype=float)
    background[:, energy_mask] = baseline_selected
    corrected[:, energy_mask] = corrected_selected
    final_weights[:, energy_mask] = weights_selected

    output_file = Path(output_file)
    csv_output_file = Path(csv_output_file)
    spectrum_text_directory = Path(spectrum_text_directory)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    csv_output_file.parent.mkdir(parents=True, exist_ok=True)
    spectrum_text_directory.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        energy=energy,
        original=intensities,
        background=background,
        corrected=corrected,
        selected_lambda=selected_lambda,
        lambda_grid=np.asarray(diagnostics["lambda_grid"]),
        joint_log_residual=np.asarray(diagnostics["joint_log_residual"]),
        joint_log_roughness=np.asarray(diagnostics["joint_log_roughness"]),
        curvature=np.asarray(diagnostics["curvature"]),
        energy_min=energy_min,
        energy_max=energy_max,
        arpls_ratio=ratio,
        arpls_max_iter=max_iter,
        labels=np.asarray(labels, dtype=str),
        final_weights=final_weights,
    )

    with csv_output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "lambda",
                "joint_log_residual",
                "joint_log_roughness",
                "curvature",
                "number_converged",
            ]
        )
        for row in zip(
            np.asarray(diagnostics["lambda_grid"]),
            np.asarray(diagnostics["joint_log_residual"]),
            np.asarray(diagnostics["joint_log_roughness"]),
            np.asarray(diagnostics["curvature"]),
            np.asarray(diagnostics["number_converged"]),
        ):
            writer.writerow(row)

    for index, label in enumerate(labels):
        output_table = np.column_stack(
            (
                energy,
                intensities[index],
                background[index],
                corrected[index],
                final_weights[index],
            )
        )
        np.savetxt(
            spectrum_text_directory / f"spectrum_{index:03d}.txt",
            output_table,
            header="Energy_meV Original Background Corrected FinalWeight",
        )

    return background, corrected, final_weights


def _print_summary(
    labels: list[str],
    energy_selected: np.ndarray,
    selected_lambda: float,
    diagnostics: dict[str, object],
    final_infos: list[dict[str, float | int | bool | str]],
) -> None:
    selection_info = dict(diagnostics["selection_info"])
    print(f"Number of spectra: {len(labels)}")
    print(
        "Selected energy interval: "
        f"{energy_selected[0]:.8g} to {energy_selected[-1]:.8g} meV"
    )
    print(f"Number of selected energy points: {energy_selected.size}")
    print(f"Selected joint lambda: {selected_lambda:.8g}")
    maximum_curvature = float(selection_info["maximum_curvature"])
    if np.isfinite(maximum_curvature):
        print(f"Maximum L-curve curvature: {maximum_curvature:.8g}")
    else:
        print("Maximum L-curve curvature: not evaluated or non-finite")
    print(f"Reliable selection: {selection_info['selection_reliable']}")
    print("arPLS convergence for each spectrum:")
    for label, info in zip(labels, final_infos):
        print(
            f"  {label}: converged={info['converged']}, "
            f"iterations={info['iterations']}, "
            f"relative_weight_change={float(info['relative_weight_change']):.3e}, "
            f"status={info['status']}"
        )
    print("Warnings:")
    selection_warnings = list(selection_info["warnings"])
    final_fit_warnings = [
        f"Final arPLS fit did not converge for {label}: {info['status']}."
        for label, info in zip(labels, final_infos)
        if not bool(info["converged"])
    ]
    all_warnings = selection_warnings + final_fit_warnings
    if all_warnings:
        for message in all_warnings:
            print(f"  - {message}")
    else:
        print("  None")


def run_synthetic_test() -> dict[str, object]:
    """Run focused, deterministic checks of selection, scaling, and safeguards."""
    random_generator = np.random.default_rng(20260905)
    energy = np.linspace(5.0, 200.0, 240)
    normalized_energy = (energy - energy.min()) / np.ptp(energy)
    background_shape = 0.15 + 0.12 * normalized_energy + 0.04 * normalized_energy**2
    peak_shape = (
        0.75 * np.exp(-0.5 * ((energy - 45.0) / 4.0) ** 2)
        + 0.48 * np.exp(-0.5 * ((energy - 105.0) / 7.0) ** 2)
        + 0.30 * np.exp(-0.5 * ((energy - 158.0) / 5.0) ** 2)
    )
    overall_scales = np.array([0.5, 3.0, 25.0])
    spectra = np.vstack(
        [
            scale
            * (
                background_shape
                + peak_shape
                + random_generator.normal(0.0, 0.008, energy.size)
            )
            for scale in overall_scales
        ]
    )
    original_copy = spectra.copy()
    test_grid = np.logspace(2, 8, 25)

    selected_lambda, diagnostics = select_joint_lambda(
        spectra,
        test_grid,
        ratio=1e-6,
        max_iter=150,
    )
    assert np.isfinite(selected_lambda) and selected_lambda > 0
    baselines, corrected, weights, infos = apply_common_lambda(
        spectra,
        selected_lambda,
        ratio=1e-6,
        max_iter=150,
    )
    assert all(float(info["lambda"]) == selected_lambda for info in infos)
    assert baselines.shape == spectra.shape
    assert corrected.shape == spectra.shape
    assert weights.shape == spectra.shape
    assert np.any(corrected < 0), "Quantitative corrected data appear clipped."
    assert np.array_equal(spectra, original_copy), "Input spectra were modified."

    # Multiplying a spectrum by a large constant should not change its own
    # normalized L-curve measures, demonstrating protection from scale dominance.
    reference_evaluation = evaluate_lambda_candidate(
        spectra,
        selected_lambda,
        ratio=1e-6,
        max_iter=150,
    )
    rescaled_spectra = spectra.copy()
    rescaled_spectra[-1] *= 1e4
    rescaled_evaluation = evaluate_lambda_candidate(
        rescaled_spectra,
        selected_lambda,
        ratio=1e-6,
        max_iter=150,
    )
    assert np.allclose(
        np.asarray(reference_evaluation["residual_measures"])[-1],
        np.asarray(rescaled_evaluation["residual_measures"])[-1],
        rtol=2e-4,
        atol=1e-10,
    )
    assert np.allclose(
        np.asarray(reference_evaluation["roughness_measures"])[-1],
        np.asarray(rescaled_evaluation["roughness_measures"])[-1],
        rtol=2e-4,
        atol=1e-10,
    )

    # Exercise explicit failure and reliability-warning paths.
    _, _, constant_info = arpls(np.ones(20), lam=1e4)
    assert not constant_info["converged"]
    assert "negative_residual" in str(constant_info["status"])
    _, warning_messages = _assess_selection_reliability(
        lambda_grid=np.array([1e2, 1e3, 1e4, 1e5]),
        joint_log_residual=np.array([0.0, 0.0, 0.0, 0.0]),
        joint_log_roughness=np.array([0.0, 0.0, 0.0, 0.0]),
        curvature=np.array([np.nan, 0.0, 0.0, np.nan]),
        number_converged=np.array([0, 0, 0, 0]),
        number_spectra=3,
        selected_index=1,
        neighbor_changes={"smaller": 0.0, "larger": 0.0},
    )
    assert any("boundary" in message for message in warning_messages)
    assert any("did not converge" in message for message in warning_messages)
    assert any("constant" in message for message in warning_messages)

    print("Synthetic joint-arPLS test passed.")
    print(f"Synthetic selected lambda: {selected_lambda:.8g}")
    return {
        "selected_lambda": selected_lambda,
        "diagnostics": diagnostics,
        "baselines": baselines,
        "corrected": corrected,
        "weights": weights,
        "infos": infos,
    }


def main() -> None:
    energy, intensities, labels = load_datasets(
        datasets=DATASETS,
        npz_file=NPZ_INPUT_FILE,
        stacked_npy_file=STACKED_NPY_INPUT_FILE,
        base_directory=SCRIPT_DIRECTORY,
        grid_rtol=ENERGY_GRID_RTOL,
        grid_atol=ENERGY_GRID_ATOL,
    )
    original_snapshot = intensities.copy()
    energy_mask, energy_selected, intensities_selected = select_energy_range(
        energy,
        intensities,
        ENERGY_MIN,
        ENERGY_MAX,
    )

    if MANUAL_LAMBDA is None:
        selected_lambda, diagnostics = select_joint_lambda(
            intensities_selected,
            LAMBDA_GRID,
            ratio=ARPLS_RATIO,
            max_iter=ARPLS_MAX_ITER,
        )
    else:
        if not np.isfinite(MANUAL_LAMBDA) or MANUAL_LAMBDA <= 0:
            raise ValueError("MANUAL_LAMBDA must be None or a finite positive number.")
        selected_lambda = float(MANUAL_LAMBDA)
        diagnostics = _manual_diagnostics(LAMBDA_GRID, selected_lambda)

    baseline_selected, corrected_selected, weights_selected, final_infos = (
        apply_common_lambda(
            intensities_selected,
            selected_lambda=selected_lambda,
            ratio=ARPLS_RATIO,
            max_iter=ARPLS_MAX_ITER,
        )
    )
    if not np.array_equal(intensities, original_snapshot):
        raise RuntimeError("Internal error: the input intensity array was modified.")

    save_results(
        output_file=OUTPUT_FILE,
        csv_output_file=CSV_OUTPUT_FILE,
        spectrum_text_directory=SPECTRUM_TEXT_DIRECTORY,
        energy=energy,
        intensities=intensities,
        energy_mask=energy_mask,
        baseline_selected=baseline_selected,
        corrected_selected=corrected_selected,
        weights_selected=weights_selected,
        selected_lambda=selected_lambda,
        diagnostics=diagnostics,
        labels=labels,
        energy_min=ENERGY_MIN,
        energy_max=ENERGY_MAX,
        ratio=ARPLS_RATIO,
        max_iter=ARPLS_MAX_ITER,
    )

    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if MANUAL_LAMBDA is None:
        joint_figure, _ = plot_joint_diagnostics(diagnostics)
        joint_figure.savefig(
            PLOT_DIRECTORY / "joint_lambda_selection.png",
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

    spectrum_figures = plot_spectrum_results(
        energy_selected,
        intensities_selected,
        baseline_selected,
        corrected_selected,
        selected_lambda,
        labels,
    )
    for index, figure in enumerate(spectrum_figures):
        figure.savefig(
            PLOT_DIRECTORY / f"spectrum_{index:03d}_background_subtraction.png",
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

    if not 0 <= DIAGNOSTIC_SPECTRUM_INDEX < intensities.shape[0]:
        raise IndexError(
            "DIAGNOSTIC_SPECTRUM_INDEX must be between 0 and "
            f"{intensities.shape[0] - 1}."
        )
    candidate_figure, _ = plot_candidate_baselines(
        energy_selected,
        intensities_selected[DIAGNOSTIC_SPECTRUM_INDEX],
        np.asarray(LAMBDA_GRID, dtype=float),
        selected_lambda,
        ratio=ARPLS_RATIO,
        max_iter=ARPLS_MAX_ITER,
        label=labels[DIAGNOSTIC_SPECTRUM_INDEX],
    )
    candidate_figure.savefig(
        PLOT_DIRECTORY / "neighboring_lambda_baselines.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )

    _print_summary(
        labels,
        energy_selected,
        selected_lambda,
        diagnostics,
        final_infos,
    )
    print(f"Quantitative NPZ: {OUTPUT_FILE}")
    print(f"Lambda CSV: {CSV_OUTPUT_FILE}")
    print(f"Spectrum text files: {SPECTRUM_TEXT_DIRECTORY}")
    print(f"PNG figures: {PLOT_DIRECTORY}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the deterministic synthetic verification instead of loading data",
    )
    arguments = argument_parser.parse_args()
    if arguments.self_test:
        run_synthetic_test()
    else:
        main()
