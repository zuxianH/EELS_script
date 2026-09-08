#!/usr/bin/env python3
"""Subtract smooth arPLS backgrounds from momentum-integrated EELS spectra.

The input array must have shape ``(n_spectra, n_energy, 2)``.  Column 0 is
the energy axis in meV and column 1 is the intensity.  Only the selected
energy interval is fitted; the input array itself is never modified.  The
quantitative corrected intensity is constrained to be non-negative.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import MatrixRankWarning, spsolve


# ---------------------------------------------------------------------------
# User-adjustable parameters
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIRECTORY / "SiC_eels_thicknessdata_center_40px_new.npy"
OUTPUT_NPY_FILE = (
    SCRIPT_DIRECTORY / "SiC_eels_thicknessdata_center_40px_new_arpls.npy"
)
OUTPUT_NPZ_FILE = (
    SCRIPT_DIRECTORY / "SiC_eels_thicknessdata_center_40px_new_arpls_full.npz"
)
PLOT_DIRECTORY = SCRIPT_DIRECTORY / "arpls_plots"

ENERGY_MIN = 5.0
ENERGY_MAX = 250.0  # meV

ARPLS_LAMBDA = 1e5
ARPLS_RATIO = 1e-6
ARPLS_MAX_ITER = 100

# Optional Gaussian smoothing before arPLS background estimation, measured in
# selected-grid sample points.  Use 0.0 to disable it; values around 1.0-2.0
# are typical starting points.  The resulting background is subtracted from
# the original, unblurred spectrum so spectral peak resolution is preserved.
GAUSSIAN_BLUR_SIGMA = 0.0

# Smaller lambda -> the baseline is more flexible.
# Larger lambda  -> the baseline is smoother.
TEST_LAMBDAS = [
    1e3,
    1e4,
    1e5,
    1e6,
    1e7,
    1e8,
]
DIAGNOSTIC_SPECTRUM_INDEX = 0

# Set this to a list such as ["10 nm", "20 nm", ...] to use physical labels.
# Its length must equal the number of spectra.  None uses spectrum indices.
SPECTRUM_LABELS: list[str] | None = None

SAVE_PLOTS = True
SHOW_PLOTS = True


def load_spectra(input_file: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate an ``(n_spectra, n_energy, 2)`` EELS array."""
    data = np.load(input_file)

    if data.ndim != 3 or data.shape[2] != 2:
        raise ValueError(
            "Expected data with shape (n_spectra, n_energy, 2), "
            f"but received {data.shape}."
        )
    if data.shape[0] < 1:
        raise ValueError("The input must contain at least one spectrum.")
    if data.shape[1] < 3:
        raise ValueError("Each spectrum must contain at least three energy points.")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"Expected numeric data, but received dtype {data.dtype}.")

    energy = data[0, :, 0]
    intensities = data[:, :, 1]

    if not np.all(np.isfinite(energy)):
        raise ValueError("The energy axis contains non-finite values.")
    if not np.all(np.isfinite(intensities)):
        raise ValueError("The intensity data contain non-finite values.")

    for i in range(data.shape[0]):
        if not np.allclose(data[i, :, 0], energy):
            raise ValueError(f"Spectrum {i} does not share the common energy axis.")

    return data, energy, intensities


def select_energy_range(
    energy: np.ndarray,
    intensities: np.ndarray,
    energy_min: float,
    energy_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select an inclusive fitting interval without altering the full arrays."""
    if not np.isfinite(energy_min) or not np.isfinite(energy_max):
        raise ValueError("Energy limits must be finite.")
    if energy_min > energy_max:
        raise ValueError("ENERGY_MIN must not be greater than ENERGY_MAX.")

    energy_mask = (energy >= energy_min) & (energy <= energy_max)
    if np.count_nonzero(energy_mask) < 3:
        raise ValueError(
            "The selected energy interval must contain at least three points; "
            f"it contains {np.count_nonzero(energy_mask)}."
        )

    energy_selected = energy[energy_mask]
    intensity_selected = intensities[:, energy_mask]
    return energy_mask, energy_selected, intensity_selected


def gaussian_blur_spectra(
    intensities: np.ndarray,
    sigma: float = 0.0,
) -> np.ndarray:
    """Return spectra optionally Gaussian-smoothed along the energy axis.

    ``sigma`` is measured in energy-grid sample points.  A value of zero
    disables smoothing and still returns a floating-point copy, ensuring the
    caller's input is never modified.
    """
    intensities = np.asarray(intensities, dtype=float)
    if intensities.ndim not in (1, 2):
        raise ValueError(
            "Gaussian blur expects a 1D spectrum or a 2D "
            "(n_spectra, n_energy) array."
        )
    if not np.all(np.isfinite(intensities)):
        raise ValueError("Cannot Gaussian-blur non-finite intensity values.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("GAUSSIAN_BLUR_SIGMA must be finite and non-negative.")
    if sigma == 0:
        return intensities.copy()
    axis = 0 if intensities.ndim == 1 else 1
    return gaussian_filter1d(
        intensities,
        sigma=float(sigma),
        axis=axis,
        mode="nearest",
    )


def arpls(
    y: np.ndarray,
    lam: float = 1e5,
    ratio: float = 1e-6,
    max_iter: int = 100,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """
    Estimate the smooth background of a 1D spectrum using arPLS.

    Parameters
    ----------
    y : 1D numpy.ndarray
        Intensity spectrum.
    lam : float
        Smoothness penalty.  Smaller values give a more flexible baseline;
        larger values give a smoother baseline.
    ratio : float
        Convergence threshold for relative weight change.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    baseline : 1D numpy.ndarray
        Estimated background.
    info : dict
        Convergence information.  In addition to the requested fields, the
        dictionary includes ``termination_reason`` for diagnostics.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"arpls expects a 1D spectrum, but received shape {y.shape}.")
    if y.size < 3:
        raise ValueError("arpls requires at least three intensity points.")
    if not np.all(np.isfinite(y)):
        raise ValueError("The spectrum contains non-finite intensity values.")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a finite positive number.")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be a finite positive number.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")

    n_points = y.size
    diagonal_length = n_points - 2
    difference_matrix = sparse.diags(
        diagonals=(
            np.ones(diagonal_length),
            -2.0 * np.ones(diagonal_length),
            np.ones(diagonal_length),
        ),
        offsets=(0, 1, 2),
        shape=(diagonal_length, n_points),
        format="csc",
    )
    smoothness_penalty = lam * (difference_matrix.T @ difference_matrix)

    weights = np.ones(n_points, dtype=float)
    baseline = np.zeros_like(y)
    relative_change = np.inf
    converged = False
    termination_reason = "maximum_iterations_reached"

    for iteration in range(1, max_iter + 1):
        weight_matrix = sparse.diags(weights, offsets=0, format="csc")
        system_matrix = weight_matrix + smoothness_penalty

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                baseline = spsolve(system_matrix, weights * y)
        except (MatrixRankWarning, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"The sparse arPLS baseline solve failed at iteration {iteration}."
            ) from exc

        if not np.all(np.isfinite(baseline)):
            raise RuntimeError(
                f"The arPLS solve produced a non-finite baseline at iteration {iteration}."
            )

        residual = y - baseline
        negative_residuals = residual[residual < 0]

        if negative_residuals.size < 2:
            termination_reason = "too_few_negative_residuals"
            break

        negative_mean = float(np.mean(negative_residuals))
        negative_std = float(np.std(negative_residuals))
        if not np.isfinite(negative_mean) or not np.isfinite(negative_std):
            termination_reason = "non_finite_negative_residual_statistics"
            break

        scale = max(1.0, float(np.max(np.abs(negative_residuals))))
        if negative_std <= np.finfo(float).eps * scale:
            termination_reason = "near_zero_negative_residual_std"
            break

        # Standard arPLS asymmetric logistic reweighting.  Clipping the
        # exponent prevents overflow without changing the practical weights.
        exponent = 2.0 * (
            residual - (2.0 * negative_std - negative_mean)
        ) / negative_std
        exponent = np.clip(exponent, -60.0, 60.0)
        new_weights = 1.0 / (1.0 + np.exp(exponent))

        if not np.all(np.isfinite(new_weights)):
            termination_reason = "non_finite_weights"
            break

        weight_norm = np.linalg.norm(weights)
        if not np.isfinite(weight_norm) or weight_norm <= np.finfo(float).tiny:
            termination_reason = "invalid_weight_norm"
            break

        relative_change = float(np.linalg.norm(weights - new_weights) / weight_norm)
        if not np.isfinite(relative_change):
            termination_reason = "non_finite_relative_weight_change"
            break

        weights = new_weights
        if relative_change < ratio:
            converged = True
            termination_reason = "converged"
            break

    info: dict[str, float | int | bool | str] = {
        "iterations": iteration,
        "converged": converged,
        "relative_weight_change": relative_change,
        "termination_reason": termination_reason,
    }
    return baseline, info


def apply_arpls_to_spectra(
    intensity_selected: np.ndarray,
    lam: float,
    ratio: float,
    max_iter: int,
    gaussian_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | bool | str]]]:
    """Fit every spectrum and constrain corrected intensity to be non-negative.

    Gaussian smoothing, when enabled, is used only to estimate the background.
    That background is subtracted from the original selected intensity before
    negative corrected values are clipped to zero.
    """
    intensity_selected = np.asarray(intensity_selected, dtype=float)
    if intensity_selected.ndim != 2:
        raise ValueError(
            "intensity_selected must have shape (n_spectra, n_selected_energy)."
        )
    if not np.all(np.isfinite(intensity_selected)):
        raise ValueError("intensity_selected contains non-finite values.")

    fit_intensity_selected = gaussian_blur_spectra(
        intensity_selected,
        sigma=gaussian_sigma,
    )

    background_selected = np.zeros_like(intensity_selected, dtype=float)
    corrected_selected = np.zeros_like(intensity_selected, dtype=float)
    infos: list[dict[str, float | int | bool | str]] = []

    for i in range(intensity_selected.shape[0]):
        baseline, info = arpls(
            fit_intensity_selected[i],
            lam=lam,
            ratio=ratio,
            max_iter=max_iter,
        )
        unconstrained_corrected = intensity_selected[i] - baseline
        negative_points_clipped = int(np.count_nonzero(unconstrained_corrected < 0))
        background_selected[i] = baseline
        corrected_selected[i] = np.maximum(unconstrained_corrected, 0.0)
        info = dict(info)
        info["negative_points_clipped"] = negative_points_clipped
        info["minimum_before_clipping"] = float(np.min(unconstrained_corrected))
        infos.append(info)

    return background_selected, corrected_selected, infos


def plot_spectrum_result(
    energy: np.ndarray,
    intensity: np.ndarray,
    background: np.ndarray,
    corrected: np.ndarray,
    spectrum_index: int,
    label: str | None = None,
    fit_intensity: np.ndarray | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the original, fitted-background, and corrected curves."""
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(energy, intensity, label="Original")
    if fit_intensity is not None:
        axes.plot(
            energy,
            fit_intensity,
            linestyle=":",
            linewidth=1.2,
            label="Gaussian-blurred fit input",
        )
    axes.plot(energy, background, label="arPLS background")
    axes.plot(energy, corrected, label="Background subtracted (non-negative)")
    axes.set_xlabel("Energy loss (meV)")
    axes.set_ylabel("Intensity")

    spectrum_name = str(spectrum_index) if label is None else label
    axes.set_title(f"Spectrum {spectrum_name}: arPLS background subtraction")
    axes.legend()
    axes.grid(alpha=0.2)
    figure.tight_layout()
    return figure, axes


def plot_lambda_diagnostics(
    energy: np.ndarray,
    intensity: np.ndarray,
    lambdas: list[float] | tuple[float, ...] | np.ndarray,
    ratio: float,
    max_iter: int,
    gaussian_sigma: float = 0.0,
) -> tuple[plt.Figure, list[dict[str, float | int | bool | str]]]:
    """Plot one spectrum with the background fitted at each candidate lambda."""
    figure, axes = plt.subplots(figsize=(9, 5.5))
    axes.plot(energy, intensity, color="black", linewidth=1.5, label="Original")
    fit_intensity = gaussian_blur_spectra(intensity, sigma=gaussian_sigma)
    if gaussian_sigma > 0:
        axes.plot(
            energy,
            fit_intensity,
            color="0.45",
            linestyle=":",
            linewidth=1.2,
            label=fr"Gaussian-blurred input ($\sigma={gaussian_sigma:g}$ points)",
        )

    infos: list[dict[str, float | int | bool | str]] = []
    for lam in lambdas:
        baseline, info = arpls(
            fit_intensity,
            lam=float(lam),
            ratio=ratio,
            max_iter=max_iter,
        )
        axes.plot(energy, baseline, label=fr"$\lambda={lam:.0e}$")
        infos.append(info)

    axes.set_xlabel("Energy loss (meV)")
    axes.set_ylabel("Intensity")
    axes.set_title("arPLS lambda diagnostic")
    axes.legend()
    axes.grid(alpha=0.2)
    figure.tight_layout()
    return figure, infos


def save_results(
    data: np.ndarray,
    energy: np.ndarray,
    intensities: np.ndarray,
    energy_mask: np.ndarray,
    background_selected: np.ndarray,
    corrected_selected: np.ndarray,
    output_npy_file: str | Path,
    output_npz_file: str | Path,
    energy_min: float,
    energy_max: float,
    arpls_lambda: float,
    arpls_ratio: float,
    arpls_max_iter: int,
    gaussian_blur_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Save the convenient NPY and preferred, analysis-safe NPZ outputs."""
    n_spectra = data.shape[0]

    background = np.full_like(intensities, np.nan, dtype=float)
    corrected = np.full_like(intensities, np.nan, dtype=float)
    background[:, energy_mask] = background_selected
    corrected[:, energy_mask] = corrected_selected

    # In this convenience format, intensities outside the fitted interval are
    # deliberately left unchanged.  Use the NPZ output for quantitative work.
    corrected_data = data.copy()
    for i in range(n_spectra):
        corrected_data[i, energy_mask, 1] = corrected_selected[i]

    np.save(output_npy_file, corrected_data)
    np.savez_compressed(
        output_npz_file,
        energy=energy,
        original=intensities,
        background=background,
        corrected=corrected,
        energy_min=energy_min,
        energy_max=energy_max,
        arpls_lambda=arpls_lambda,
        arpls_ratio=arpls_ratio,
        arpls_max_iter=arpls_max_iter,
        gaussian_blur_sigma=gaussian_blur_sigma,
        corrected_is_nonnegative=True,
    )
    return corrected_data, background, corrected


def _validate_labels(labels: list[str] | None, n_spectra: int) -> None:
    if labels is not None and len(labels) != n_spectra:
        raise ValueError(
            f"SPECTRUM_LABELS has {len(labels)} entries, but there are "
            f"{n_spectra} spectra."
        )


def main() -> None:
    data, energy, intensities = load_spectra(INPUT_FILE)
    n_spectra = data.shape[0]
    _validate_labels(SPECTRUM_LABELS, n_spectra)

    energy_mask, energy_selected, intensity_selected = select_energy_range(
        energy,
        intensities,
        ENERGY_MIN,
        ENERGY_MAX,
    )
    background_selected, corrected_selected, infos = apply_arpls_to_spectra(
        intensity_selected,
        lam=ARPLS_LAMBDA,
        ratio=ARPLS_RATIO,
        max_iter=ARPLS_MAX_ITER,
        gaussian_sigma=GAUSSIAN_BLUR_SIGMA,
    )
    fit_intensity_selected = gaussian_blur_spectra(
        intensity_selected,
        sigma=GAUSSIAN_BLUR_SIGMA,
    )

    for i, info in enumerate(infos):
        print(
            f"Spectrum {i}: iterations={info['iterations']}, "
            f"converged={info['converged']}, "
            f"relative_weight_change={info['relative_weight_change']:.3e}, "
            f"reason={info['termination_reason']}, "
            f"negative_points_clipped={info['negative_points_clipped']}"
        )
        if not info["converged"]:
            warnings.warn(
                f"arPLS did not converge for spectrum {i}: "
                f"{info['termination_reason']}",
                RuntimeWarning,
                stacklevel=1,
            )

    save_results(
        data=data,
        energy=energy,
        intensities=intensities,
        energy_mask=energy_mask,
        background_selected=background_selected,
        corrected_selected=corrected_selected,
        output_npy_file=OUTPUT_NPY_FILE,
        output_npz_file=OUTPUT_NPZ_FILE,
        energy_min=ENERGY_MIN,
        energy_max=ENERGY_MAX,
        arpls_lambda=ARPLS_LAMBDA,
        arpls_ratio=ARPLS_RATIO,
        arpls_max_iter=ARPLS_MAX_ITER,
        gaussian_blur_sigma=GAUSSIAN_BLUR_SIGMA,
    )

    if SAVE_PLOTS:
        PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for i in range(n_spectra):
        label = None if SPECTRUM_LABELS is None else SPECTRUM_LABELS[i]
        figure, _ = plot_spectrum_result(
            energy_selected,
            intensity_selected[i],
            background_selected[i],
            corrected_selected[i],
            spectrum_index=i,
            label=label,
            fit_intensity=(
                fit_intensity_selected[i] if GAUSSIAN_BLUR_SIGMA > 0 else None
            ),
        )
        if SAVE_PLOTS:
            figure.savefig(
                PLOT_DIRECTORY / f"spectrum_{i}_arpls.png",
                dpi=200,
                bbox_inches="tight",
            )

    if not 0 <= DIAGNOSTIC_SPECTRUM_INDEX < n_spectra:
        raise IndexError(
            "DIAGNOSTIC_SPECTRUM_INDEX must be in the range "
            f"0 to {n_spectra - 1}."
        )

    diagnostic_figure, diagnostic_infos = plot_lambda_diagnostics(
        energy_selected,
        intensity_selected[DIAGNOSTIC_SPECTRUM_INDEX],
        TEST_LAMBDAS,
        ratio=ARPLS_RATIO,
        max_iter=ARPLS_MAX_ITER,
        gaussian_sigma=GAUSSIAN_BLUR_SIGMA,
    )
    diagnostic_figure.axes[0].set_title(
        f"Spectrum {DIAGNOSTIC_SPECTRUM_INDEX}: arPLS lambda diagnostic"
    )
    if SAVE_PLOTS:
        diagnostic_figure.savefig(
            PLOT_DIRECTORY
            / f"spectrum_{DIAGNOSTIC_SPECTRUM_INDEX}_lambda_diagnostic.png",
            dpi=200,
            bbox_inches="tight",
        )

    for lam, info in zip(TEST_LAMBDAS, diagnostic_infos):
        if not info["converged"]:
            warnings.warn(
                f"Lambda diagnostic did not converge for lambda={lam:.0e}: "
                f"{info['termination_reason']}",
                RuntimeWarning,
                stacklevel=1,
            )

    print(f"Saved convenience output: {OUTPUT_NPY_FILE}")
    print(f"Saved preferred quantitative output: {OUTPUT_NPZ_FILE}")
    print(
        "Gaussian blur before background estimation: "
        + (
            f"sigma={GAUSSIAN_BLUR_SIGMA:g} sample points"
            if GAUSSIAN_BLUR_SIGMA > 0
            else "disabled"
        )
    )
    print("Non-negative corrected-intensity constraint: enabled")
    if SAVE_PLOTS:
        print(f"Saved plots: {PLOT_DIRECTORY}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
