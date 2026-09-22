# EELS Studio

A local browser app for exploring vibrational EELS: select `.npy` diffraction scans, position a circular detector, compare spectra, subtract backgrounds, and build 2D/angle-resolved maps. Calculations follow the lab's STEM-EELS notebook workflow, reimplemented in `eels_core.py`.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open http://localhost:8501 (binds to localhost only). Once `.venv` exists, `./run_app.sh` starts it from anywhere.

## Usage

- **Load scans** — click **Browse folders** in the sidebar or type a data-folder path, then select files. The folder-selection window opens on the computer running the app; an in-app browser appears if a desktop window is unavailable. Cancelling keeps the current folder. Supports 3D `(energy, px, py)` and 6D `(dummy, energy, probe_x, probe_y, px, py)` arrays.
- **Detector** — set `px`, `py`, and `radius` under the detector image, or click the image to position the center. For 6D scans, pick probe positions in the sidebar.
- **Spectrum** — choose **Linear**, **log10**, or **Intensity × E²**, set axis limits, customize line styles, and export as CSV/NumPy/SVG/PNG. The E² option uses energy in meV and applies to plots and figure downloads; data exports retain unscaled linear intensities. Changing the display mode resets the vertical range while retaining the horizontal view. Drag to pan, Shift+drag an axis to scale it, scroll to zoom, and use the legend to hide/isolate curves.
- **Background subtraction** — in the **Background** tab, fit with arPLS, SNIP, or one of four analytic models, preview, then apply. Switch the Spectra view between Input/Corrected.
- **2D scan maps** — under **Visualization → 2D scan map**, request one or more energies to see detector-summed intensity across probe positions.
- **Angle-resolved EELS** — draw a rectangle on the diffraction image to sum a strip across one detector direction, producing an energy-vs-pixel map.

## Tests

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
```

Browser-gesture tests are optional and need Playwright plus a browser:

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install firefox
EELS_BROWSER=firefox .venv/bin/python -m pytest -q tests/test_axis_browser.py tests/test_angle_browser.py tests/test_background_browser.py
```

## License

MIT — see [LICENSE](LICENSE).
