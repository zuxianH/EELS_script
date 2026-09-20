# EELS Studio

A local browser app for selecting `.npy` diffraction scans, adjusting a circular detector, and comparing EELS spectra. The calculations follow `STEM-EELS.ipynb`; the notebook and original data are unchanged.

## Run

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open http://localhost:8501. Stop the server with Ctrl+C. The server binds to your local computer only. If `.venv` is already installed, run `./run_app.sh` to start the app. The launcher also works when invoked from another directory.

## Use

1. Enter a local data folder and select any number of scans. The first two supported files are selected initially. Temperature labels such as `300 K` and `1000 K` are inferred from filenames where available. Edit labels in **Curve labels**.
2. Set the detector radius and center offsets in pixels, or open **Detector preview** and click the diffraction image to position the center. The spectrum and sidebar offsets update immediately. On the preview, **px is vertical and py is horizontal**, matching the notebook's array convention. **Reset detector center** returns to the array center. Clicking uses the same offsets for every selected scan; a position outside another selected plane is rejected with an explanation. For 6D scans, use **Probe positions (x, y)** to select individual coordinate pairs, such as `(0, 0)`, `(15, 15)`, and `(29, 29)`. Each pair produces one spectrum per file; the first sample (index 0) is used automatically.
3. Confirm **Simulation time step** and **Sampling stride**. Defaults are 2.5 fs and 3, from the notebook helper functions, but the notebook's STO example uses 5 fs. These values cannot be recovered reliably from a bare `.npy` file. The bin count comes from the array, not its filename.
4. Switch between linear intensity and log10, set plot limits, and inspect the detector overlay at any energy. Turn off **Show hover details** above the spectrum plot to hide the popup when comparing many curves. Under **Line appearance**, select a spectrum and customize its color, line style (solid, dashed, dotted, or dash-dot), and width. Styles are retained during the session when you switch files or rename labels, apply to SVG/PNG downloads, and are included in NPZ curve metadata. Changes update automatically; extracted spectra are cached.
5. Download CSV, NumPy data, or SVG/PNG figures from **Export spectra & figures**.

On the spectrum chart, drag an axis to move its range. Hold **Shift** and drag on either axis to scale about its current midpoint: **up/right zooms in**, **down/left zooms out**. The other axis stays unchanged. Zoom and axis positions stay in place when you change detector radius or other processing/display parameters. Editing the numeric energy or intensity limits replaces the view of that axis; the plot toolbar's **Reset axes** control restores the default view. Drag inside the plot for box zoom, or double-click to reset. These interactions only change the browser view; exported figures use the numeric display controls.

### 2D scan maps

Select **Visualization → 2D scan map**, choose a selected 6D file under **Map scan**, and set the energies and circular detector. The first sample (index 0) is used automatically. Defaults match the BTO notebook: requested energy **60 meV**, radius **21 pixels**, centered detector, time step **5 fs**, stride **3**, and FFT-shifted energy ordering. Enter one or more values under **Map energies (meV)**, for example `20, 40, 60, 80`. Maps appear in a grid with up to three columns, using the nearest recorded bins. Each panel shows its actual energy and array index; requests that select the same bin share one panel.

Each map sums detector pixels at every probe position and uses raw intensities with a linear color scale. **Shared color scale** starts enabled so colors represent the same intensity across energies. Turn it off to inspect each map on its own intensity scale. Horizontal is probe x and vertical is probe y. Hover to inspect an intensity; download the figure as PNG, the `(probe_x, probe_y)` array as `.npy`, or the array plus calibration and detector metadata as `.npz`. For multiple energies, **Download all maps** exports a combined PNG or an NPZ containing `scan_maps` with shape `(energy_map, probe_x, probe_y)`, `selected_energies_mev`, `energy_indices`, and JSON settings. Maps follow the requested energy order after merging duplicate bins. All-zero slices are displayed with an explanation.

Map controls are independent of spectrum controls. Spectrum normalization and broadening do not apply. The map view does not extract spectra first, so a zero-sum spectrum cannot block map visualization. Switch back to **Spectra & detector** for those operations.

File inspection reads only the header. Maps, spectra, and diffraction previews use direct block reads without mapping the full file. For the 159 GiB BTO scan, a map reads one selected energy slice (about 272 MiB total) in roughly 9 MiB probe rows. Larger rows are split into blocks targeting 16 MiB; a single detector plane is the minimum block. Temporary reduction arrays add some RAM overhead. Energies are processed sequentially; only the resulting small maps/spectra/previews are cached. Reading additional energies increases disk I/O without loading the full scan into RAM. C-order and Fortran-order numeric arrays are supported; C-order scans provide the most efficient detector-row reads.

To smooth/broaden spectra, enable **Gaussian broadening → Broaden EELS spectrum** in the sidebar and enter **Gaussian σ (meV)**. This is the Gaussian standard deviation; the equivalent FWHM is displayed below it. Broadening starts disabled. It applies to linear EELS intensity after detector integration and before log display, and uses each file's energy spacing. The diffraction preview stays unchanged. All plots and downloads include broadening when enabled; turn it off to recover unbroadened spectra. NPZ metadata records sigma and the boundary settings.

The Gaussian kernel extends to four standard deviations and uses reflecting boundaries. This preserves the sum of the recorded intensities without wrapping the high-energy endpoint to the low-energy endpoint. Features close to either endpoint depend on this boundary assumption. Sigma must not exceed the recorded energy span, and broadening requires at least two bins.

Supported arrays:

- `(energy, px, py)`: one spectrum per file, including both supplied silicon arrays with shape `(600, 267, 266)`.
- `(sample, energy, probe_x, probe_y, px, py)`: one spectrum per selected `(probe_x, probe_y)` pair at the first sample.

When mixing formats, probe controls affect 6D files; each 3D file still contributes one curve. With multiple 6D files, the selector offers coordinate pairs valid in all selected files. Different energy lengths and detector-plane shapes are supported; center offsets are relative to each array's integer center. Shared time calibration must apply to all selected files. The app lists `.npy` files directly inside the chosen folder, not subfolders.

## Angle-resolved phonon EELS

Open **Angle-resolved EELS** and enable the map, then choose a file/probe under **Map spectrum**. Drag a rectangle on the diffraction image; drawing a new box replaces the previous selection. You can also move/resize the outline or enter the inclusive row/column bounds numerically. Choose the preview energy to find a useful diffraction pattern. The rectangular selection is independent of the circular detector used for 1D spectra.

**Horizontal (py)** retains detector columns and sums rows, matching `data[:, row_min:row_max+1, col_min:col_max+1].sum(axis=1)` in the supplied calculation. **Vertical (px)** retains rows and sums columns instead. The map has energy loss on the vertical axis and detector pixel offset from the array's integer center on the horizontal axis. No mrad or momentum calibration is assumed. Bounds are remembered separately for each file/plane during the session; all selected 6D probe spectra are available in the source selector.

The sidebar's energy calibration, FFT ordering, full-probe normalization, and Gaussian broadening apply to the map. Broadening runs along energy only. Integration and corrections use linear intensities; log10 affects color display only and masks nonpositive bins. The colorbar represents strip-summed intensity, not an intensity density per mrad. The diffraction image shows raw data.

Set the map energy limits and optionally its color limits (in log10 or linear display units). **Map NumPy + settings** saves `energy_mev`, `pixel_offset`, the full-range linear `intensity` array of shape `(energy, pixel)`, and `metadata_json` with the source/probe, inclusive box bounds, retained axis, and processing settings. **Map PNG/SVG** exports the displayed map at the chosen limits; PNG resolution follows the **PNG export resolution** setting at the top of the page (default 300 DPI). Ratios are not computed.

## Calculation and exports

The detector sums pixels inside a circle, including its boundary. Optional normalization divides by the sum of the entire `(energy, px, py)` probe block. This is **not** unit-area normalization of the extracted spectrum. Array intensities are assumed already FFT-shifted, matching the notebook; the ordering control also supports raw unshifted FFT output. Float64 accumulation and bounded blocks avoid copying an entire scan; floating-point results may differ from notebook reductions at rounding precision.

The energy axis uses:

```python
np.fft.fftshift(np.fft.fftfreq(n_energy, timestep_fs * stride / 1000)) * 4.13566769692386
```

CSV uses one row per curve and energy bin, with file/probe labels. NPZ contains `curve_000`, `curve_001`, etc., each with columns `[energy_meV, intensity]`, plus a non-pickled JSON string containing settings and curve provenance:

```python
import json
import numpy as np

with np.load("eels_spectra.npz", allow_pickle=False) as result:
    first_curve = result["curve_000"]
    metadata = json.loads(str(result["metadata_json"]))
```

The notebook-compatible `.npy` download has shape `(n_curves, n_energy, 2)` and is available when energy lengths match. Curves follow selected file order, then selected probe-pair order. NPZ settings record these pairs in `probe_positions_xy`, and each curve records its own `probe_x` and `probe_y`. Use NPZ or CSV to retain labels. All data exports preserve full-range **linear** intensities. Figure downloads include all curves and use the explicit plot controls, not temporary browser zoom or legend visibility. The plot toolbar can save the current browser view as SVG.

Files with unsupported shapes/dtypes appear under **unsupported files**. Object arrays are never unpickled. Detector circles extending past the recorded plane integrate only available pixels; the preview flags this. Time calibration and normalization are scientific choices: confirm them for your simulation before interpreting peak positions or intensities.

## Validation

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
```

Tests check direct reads, 2D maps, and extraction against the actual notebook functions, check 3D and 6D inputs, shifted detectors, click-coordinate conversion, Gaussian width and boundary behavior, normalization, invalid inputs, exports, and application interactions.

The optional axis-gesture browser test needs Playwright and a browser. Install with `.venv/bin/python -m pip install playwright` and `.venv/bin/python -m playwright install firefox`, then run `EELS_BROWSER=firefox .venv/bin/python -m pytest -q tests/test_axis_browser.py`. Alternatively, set `EELS_CHROME_PATH` to an installed Chrome/Chromium executable.

## Background subtraction (extracted 1D spectra)

Subtraction starts disabled. Open **Background**, choose **arPLS** (default) or
**SNIP**, set a fit domain, and press **Preview** to fit only the selected curve.
The two preview panels share their energy axis: muted input and a dashed baseline
above, corrected intensity below. Requested and actual bin-aligned fit bounds are
shown above the chart; both panels shade the actual fitted interval. Editing a fit
setting clears the old preview until **Preview** is pressed again. A new preview
updates the traces and shading while preserving your viewing zoom.
The initial preview focuses on the selected fit interval, with intensity limits
computed from samples inside that interval so an off-screen zero-loss peak does
not flatten the visible signal. **Focus fit interval** restores this view after
manual zooming; **Show full spectrum** restores the recorded domain and its
intensity range. These view controls never refit or modify the fitting bounds,
signed data, applied results, or exports. Full-domain fitting initially shows the
full recorded domain; use axis gestures to inspect a smaller region.
**Preview intensity display** defaults to **Linear**,
with signed residuals and a zero reference. Select **log10** to inspect the input,
baseline, and positive residuals over a wider dynamic range. Nonpositive samples
are masked with gaps; the zero reference is hidden because log10(0) is undefined.
Switching display preserves the fit, signed arrays, exports, and energy zoom;
intensity axes rescale for the changed units. Fitting always uses linear intensity.
Hold **Shift**
and drag either panel's axis to scale about its midpoint: **up/right zooms in**,
**down/left zooms out**. Both energy axes move together; each intensity axis scales
independently. Plain axis drags pan, and the toolbar resets axes. Preview and Spectra
keep separate viewports, retained across reruns for the selected curve. These gestures
only change the view, never the fit interval or baseline. Preview
never changes the main spectra or downloads. **Apply to all spectra** independently
fits every selected 3D-file/6D-probe curve and commits only if every fit is valid.
It selects **Corrected** in Spectra. Use **Input / Corrected** there to choose the
plotted and exported signal. **Reset background** clears results and restores Input
without changing extraction, styles, calibration, or plot limits.

Draft controls, preview results, and applied results are separate. Unapplied edits
are labelled and never change applied exports; changing controls hides a stale
preview. Changes to source revision, detector/probe selection, normalization,
calibration/ordering, or Gaussian broadening invalidate results. Settings remain
available for refitting. Adding/removing curves clears the whole applied batch;
renaming or recoloring a curve does not. Fits run only on explicit actions and are
cached on the small extracted 1D arrays. Baseline adjustments reuse extraction
caches and never read a full 6D scan. Detector previews and both map modes are
unaffected, including their exports and provenance.

### Domain and processing

**Selected energy interval** starts with editable common positive-energy coverage
when it contains at least eight bins per curve; otherwise it uses common recorded
coverage. These are initial bounds, not a zero-loss-tail exclusion: the first
positive-energy bin may still be in the tail. If no valid common interval exists,
choose compatible curves or use full recorded domains. A local interval should
include useful information about the background around the peak.

Selected bounds must be finite, ordered, and fully covered by every target curve.
Bins inside the inclusive requested bounds are fitted, rounding endpoints inward
to the recorded grid. At least **eight bins** are required. Diagnostics record
requested and actual bin-aligned bounds. No interval is silently shortened to fit
an incompatible curve. **Full recorded spectrum** uses each curve's own domain and
reports coverage differences. Domains containing zero energy produce a warning;
no zero-loss exclusion or gain/loss mirroring is performed. Intervals are contiguous;
arbitrary masked-region fitting is not supported. Fit bounds are independent of
view limits and browser zoom, so zooming onto a peak does not refit.

The deliberate processing order is:

```
detector extraction / optional full-probe normalization
→ energy ordering → optional Gaussian broadening
→ background estimation and subtraction → display transform
```

Fitting uses linear, energy-unweighted intensity **after broadening**, never log10,
normalized plot coordinates, or an energy-weighted display. Changing Gaussian
sigma requires a new fit; this ordering does not imply the operations commute.
“Normalize full probe block” retains its existing definition, dividing by the full
probe-block sum, not the detector spectrum's area. There is no renormalization after
subtraction. Input arrays and original grids are preserved, without interpolation.

Within the fitted interval, `corrected = input - baseline`. Negative residuals are
retained; neither baseline positivity nor zero-valued valleys are imposed. Outside
it, full-length baseline/corrected arrays contain NaN with a false validity mask.
There is no extrapolation, zero filling, or splicing in uncorrected intensities.
Corrected log10 plots mask nonpositive samples and leave gaps, in Plotly and
Matplotlib exports alike. Linear arrays/downloads retain signed values. Input log10
plots retain the legacy clipping behavior.

### Parameters and limitations

- **arPLS:** maintained `pybaselines.Baseline.arpls`, pinned to tested pybaselines
  1.2.1, with second-order differences. Synchronized slider/numeric controls expose
  `log10(lambda)` from 2–10, initially 5 (`lambda=1e5`). Larger lambda gives a
  smoother baseline. This is a starting point, not a TACAW optimum. Advanced
  tolerance and maximum iterations initially equal `1e-3` and `100`. Tolerance
  history, finite outputs, and library warnings determine validity. Nonconverged
  fits are rejected. Exactly constant inputs have a distinct constant-baseline,
  zero-residual status; almost-flat spectra do not use that shortcut. Different
  energy spacings mean equal numerical lambda need not give equal smoothing in
  energy units.
- **SNIP:** maintained `pybaselines.Baseline.snip` with `decreasing=True`,
  `filter_order=2`, and `smooth_half_window=None`. The maximum half-window is in
  **meV**, converted independently to the nearest bin on each grid (half ties
  rounded up). Both bins and effective meV are reported. The converted window must
  be from 1 through `(fitted_bins - 1) // 2`; incompatible requests are rejected,
  never clamped. SNIP uses linear intensity without a logarithmic transformation.
  Diagnostics report completion/window size, not an arPLS convergence statistic.
- **power / power0 / exppoly / pVoigt:** analytic models ported from
  [Vibrational-EELS_background_subtraction](https://github.com/PanGroup-UCI/Vibrational-EELS_background_subtraction)
  (MATLAB; Yan et al., *Nature* **645**, 893–899, 2025):
  `power`: `a0·x^-a1 + a2`; `power0`: `a0·x^-a1` (a third parameter is accepted for
  interface parity with `power` but has no effect, matching the MATLAB model);
  `exppoly`: `a0·exp(-a1·x + a2·x² - a3·x³) + a5`; `pVoigt`:
  `g·exp(a4·x⁴ - a2·x²) + h/(w2 + x²) + c0`; `x` is energy (meV) divided by the
  **energy scale factor**. Unlike arPLS/SNIP's single contiguous interval, these fit
  bounded nonlinear least squares (`scipy.optimize.curve_fit`, trust-region
  reflective) on 2–4 flanking **segments** that exclude the peak, then evaluate and
  subtract the background across the whole span from the first segment's start to
  the last segment's end — including the peak region between them. Segments default
  to evenly spaced windows over each selected spectrum set's common positive-energy
  coverage; `power`/`power0` start/bounds default to the toolkit's STO example,
  `exppoly`/`pVoigt` defaults are generic starting points that need tuning per
  dataset. **Auto-detect segments from peak** replaces them with windows flanking the
  preview spectrum's detected peak(s) instead: it finds local maxima on
  log(intensity) with `scipy.signal.find_peaks` (so detection is relative — at least
  a ~15% local rise — rather than compared to the domain's absolute intensity range,
  which a steeply decaying power-law/exponential background would otherwise dwarf),
  estimates each one's footprint with `scipy.signal.peak_widths`, and fills the
  widest remaining gaps between them (splitting the widest gap in half if fewer gaps
  than segments are available). It is a starting point, not a substitute for checking
  the preview plot — it can miss a peak below that threshold or pick the wrong one
  for a noisy or multi-featured spectrum; segments remain fully editable afterward.
  A pixel/spectrum whose segments contain no data, or whose solver does not
  converge or emits a warning, is rejected the same way a nonconverged arPLS fit is.
  Coefficient uncertainties are a normal-approximation 95% CI (`1.96·√diag(pcov)`),
  not bit-identical to MATLAB's t-distribution-based nonlinear-regression CI.

An empirical baseline may remove genuine broad vibrational intensity. Subtraction
is an analysis choice, not automatic identification of nonphysical background.
Check parameter and interval sensitivity. It neither corrects Fourier leakage nor
validates multiphonon intensity. Approximate synthetic recovery tests do not establish
recovery of arbitrary broad physical features.

### Background export schema 1

Without an applied background, legacy export formats and values remain unchanged.
With an applied background, CSV and NPZ explicitly record the selected **Input** or
**Corrected** signal; figures match it and label subtracted intensity. Preview
results are never exported. Partial-domain corrected notebook `.npy` downloads are
unavailable because they cannot carry mask/provenance: use NPZ, or select Input to
retain the original notebook export. Full-domain corrected `.npy` filenames contain
`corrected`.

**Background results CSV** contains one row per full energy bin, with curve/source/
probe identity, energy, signal selection, input intensity, baseline, corrected
intensity, and validity mask. Unfitted values are empty fields. The first row's
`metadata_json` field records the complete provenance. The regular signal CSV uses
one `intensity` column and the same signal/provenance fields.

**Background results NPZ** and the applied **NumPy + settings** export contain:

- `schema_version`: scalar integer, currently 1.
- `curve_000`, etc.: established two-column `[energy_meV, selected_signal]` arrays.
- `curve_000_energy`, `_input`, `_baseline`, `_corrected`, `_validity_mask` (and so
  on): independent full-length arrays, supporting unequal curve lengths.
- `metadata_json`: non-pickled JSON with exported signal, applied configuration,
  algorithm parameters, requested/actual bounds, energy spacing, solver diagnostics,
  source/probe identity, input fingerprints, extraction settings, processing order,
  fitted input stage, curve styles, and pybaselines version.

Load with `np.load(path, allow_pickle=False)`. Arrays are stored separately from
JSON-safe curve metadata. Baseline/corrected entries outside the mask are NaN.

API references: [arPLS](https://pybaselines.readthedocs.io/en/stable/generated/api/pybaselines.Baseline.arpls.html),
[SNIP](https://pybaselines.readthedocs.io/en/stable/generated/api/pybaselines.Baseline.snip.html).

Background-specific synthetic and state tests run with
`python -m pytest -q tests/test_background.py tests/test_background_analytic.py`.
With Playwright Firefox installed,
use `EELS_BROWSER=firefox python -m pytest -q tests/test_axis_browser.py tests/test_angle_browser.py tests/test_background_browser.py`
for axis gestures, rectangle selection, applying without losing zoom, and
1700/760/390-pixel layout checks. Set `EELS_BACKGROUND_SCREENSHOTS` to an existing
directory to retain desktop and narrow-screen captures.
