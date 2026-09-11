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

### 2D scan maps

Select **Visualization → 2D scan map**, choose a selected 6D file under **Map scan**, and set the energies and circular detector. The first sample (index 0) is used automatically. Defaults match the BTO notebook: requested energy **60 meV**, radius **21 pixels**, centered detector, time step **5 fs**, stride **3**, and FFT-shifted energy ordering. Enter one or more values under **Map energies (meV)**, for example `20, 40, 60, 80`. Maps appear in a grid with up to three columns, using the nearest recorded bins. Each panel shows its actual energy and array index; requests that select the same bin share one panel.

Each map sums detector pixels at every probe position and uses raw intensities with a linear color scale. **Shared color scale** starts enabled so colors represent the same intensity across energies. Turn it off to inspect each map on its own intensity scale. Horizontal is probe x and vertical is probe y. Hover to inspect an intensity; download the figure as PNG, the `(probe_x, probe_y)` array as `.npy`, or the array plus calibration and detector metadata as `.npz`. For multiple energies, **Download all maps** exports a combined PNG or an NPZ containing `scan_maps` with shape `(energy_map, probe_x, probe_y)`, `selected_energies_mev`, `energy_indices`, and JSON settings. Maps follow the requested energy order after merging duplicate bins. All-zero slices are displayed with an explanation.

Map controls are independent of spectrum controls. Spectrum normalization, detailed balance, and broadening do not apply. The map view does not extract spectra first, so a zero-sum spectrum cannot block map visualization. Switch back to **Spectra & detector** for those operations.

File inspection reads only the header. Maps, spectra, and diffraction previews use direct block reads without mapping the full file. For the 159 GiB BTO scan, a map reads one selected energy slice (about 272 MiB total) in roughly 9 MiB probe rows. Larger rows are split into blocks targeting 16 MiB; a single detector plane is the minimum block. Temporary reduction arrays add some RAM overhead. Energies are processed sequentially; only the resulting small maps/spectra/previews are cached. Reading additional energies increases disk I/O without loading the full scan into RAM. C-order and Fortran-order numeric arrays are supported; C-order scans provide the most efficient detector-row reads.

To smooth/broaden spectra, enable **Gaussian broadening → Broaden EELS spectrum** in the sidebar and enter **Gaussian σ (meV)**. This is the Gaussian standard deviation; the equivalent FWHM is displayed below it. Broadening starts disabled. It applies to linear EELS intensity after detector integration and before log display, and uses each file's energy spacing. The diffraction preview stays unchanged. All plots and downloads include broadening when enabled; turn it off to recover unbroadened spectra. NPZ metadata records sigma and the boundary settings.

To apply the temperature-dependent correction, enable **Detailed balance → Apply detailed-balance factor** in the sidebar. Set **Temperature (K)** separately for every file; names containing `_T300K_`, for example, prefill 300 K. Unrecognized temperatures must be entered manually. All probes from a file use that file's temperature. The option starts disabled and multiplies the linear spectrum by

```text
f(E,T) = βE / (1 − exp(−βE)),  β = 1/(k_B T)
k_B ≈ 0.08617333262 meV/K
```

Energy is in meV, positive energy denotes loss, and temperature must be positive. The factor at zero energy is exactly 1; evaluation is stable near zero and for large negative βE. Correction follows detector integration, optional full-probe normalization, and FFT ordering, and precedes Gaussian broadening and log display. The corrected spectrum is not renormalized. Use this option for input spectra that still require this factor. It affects every spectrum plot and download; the diffraction preview continues to show raw planes. NPZ metadata records whether correction was enabled, the formula, and each file/curve's temperature. Disable it to recover the original processing.

The Gaussian kernel extends to four standard deviations and uses reflecting boundaries. This preserves the sum of the recorded intensities without wrapping the high-energy endpoint to the low-energy endpoint. Features close to either endpoint depend on this boundary assumption. Sigma must not exceed the recorded energy span, and broadening requires at least two bins.

Supported arrays:

- `(energy, px, py)`: one spectrum per file, including both supplied silicon arrays with shape `(600, 267, 266)`.
- `(sample, energy, probe_x, probe_y, px, py)`: one spectrum per selected `(probe_x, probe_y)` pair at the first sample.

When mixing formats, probe controls affect 6D files; each 3D file still contributes one curve. With multiple 6D files, the selector offers coordinate pairs valid in all selected files. Different energy lengths and detector-plane shapes are supported; center offsets are relative to each array's integer center. Shared time calibration must apply to all selected files. The app lists `.npy` files directly inside the chosen folder, not subfolders.

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
