# EELS Studio

<!-- Use ASD-STE100-style language for README changes: short sentences, active
voice, consistent technical terms, and one instruction in each sentence. -->

EELS Studio is a local browser application for vibrational EELS analysis.
The application lets you select scans, position a circular detector, and compare spectra.
It also supports background subtraction, 2D scan maps, and angle-resolved maps.
Input formats include NumPy (`.npy`) files and local Zarr arrays.

## Start the application

Run these commands in a terminal:

```bash
cd /home/zuxian/Documents/EELS_script/EELS_studio
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_app.sh
```

Open http://localhost:8501 in a browser.
The server accepts connections from the local computer only.

After you create `.venv`, you can use the absolute path to `run_app.sh` from any directory.
You can also run `.venv/bin/python -m streamlit run app.py` from the application directory.
The application imports the local Python package directly.
No separate package installation is necessary.

See [Architecture](docs/architecture.md) for code locations and common changes.
See [Analysis workflow](docs/usage.md) for the analysis procedure.

## Load scans

1. Click **Browse folders** in the sidebar.
2. Select the data folder.
3. Select the files under **Files to compare**.

You can also enter a path in **Data folder**.
The folder selection window opens on the computer that runs the application.
If that window is not available, the application shows a folder browser.
If you cancel folder selection, the current folder does not change.

The application supports these array shapes:

- 3D: `(energy, px, py)`.
- 6D: `(dummy, energy, probe_x, probe_y, px, py)`.

## Set the detector

1. Set `px`, `py`, and `radius` below the detector image.
2. For 6D scans, select probe positions in the sidebar.

You can also click the detector image to set its center.

## Examine spectra

1. Select **Linear**, **log10**, or **Intensity × E²**.
2. Set the energy limits.
3. Set the line styles, if necessary.
4. Select an export format: CSV, NumPy, SVG, or PNG.

The E² display option uses energy in meV.
It changes plots and figure downloads.
It does not change numerical data exports.
A display mode change resets the vertical range and keeps the horizontal view.

Use these plot controls:

- Drag the plot to move the view.
- Hold **Shift** and drag an axis to change its scale.
- Scroll over the plot to change the zoom.
- Click a legend entry to hide its curve.
- Double-click a legend entry to show only its curve.
- Double-click the entry again to show all curves.

## Subtract a background

1. Open the **Background** tab.
2. Select arPLS, SNIP, or one of the four analytic models.
3. Set the fit parameters.
4. Click **Preview**.
5. Click **Apply to all spectra** after a valid preview.
6. In **Spectra**, select **Input** or **Corrected**.

## Make maps

For a 2D scan map:

1. Under **Visualization**, select **2D scan map**.
2. Select a 6D scan.
3. Enter one or more energies.

Each map shows the detector sum at each probe position.

For an angle-resolved map:

1. Open the **Angle-resolved EELS** tab.
2. Draw a rectangle on the diffraction image.
3. Select the detector direction to keep.

The application sums the strip across the other detector direction.
The result is an energy-versus-pixel map.

## Set the file order

Below **Files to compare**, drag the filenames in **File order** to set their order.
For keyboard control, select a filename.
Then press **Alt + ↑ / ↓**.

The selected files, plot curves, legend entries, and exports use this order.
Line styles stay with their files.
**Save config** saves the file order for the next session.

## Spectrum workspace layout

**Spectra** shows **Detector** on the left and **EELS Spectrum** on the right.
The panels have fixed sizes.
You can move the view and change the zoom in each plot.

Adjustable panels are disabled to prevent slow operation.
The code remains in `eels_studio/ui/components/panel_layout.py` and the adjacent `panel_layout.js`.
The control value is `ADJUSTABLE_PANELS_ENABLED = False`.
Workspace configs keep saved panel sizes and order for compatibility.
The application does not use these settings while adjustable panels are disabled.

## Save a workspace

1. Open **Workspace config** in the sidebar.
2. Click **Save config** to download `eels-studio-config.json`.

Use a separate config file for each analysis, if necessary.
Save the config again after you change settings.
The application does not save changes automatically.

The config includes these settings:

- Data folder, selected scans, probe positions, and file order.
- Curve labels, line colors, line styles, and line widths.
- Detector geometry and normalization.
- Energy calibration and Gaussian broadening.
- Intensity display, energy limits, and preview settings.
- Map settings, rectangle bounds, and export resolution.
- Background draft parameters and applied fit parameters.

The config contains paths and settings.
It does not contain scan data or cached arrays.
It does not save manual plot views, hidden legend entries, or prepared Zarr caches.

## Load a workspace

1. Open **Workspace config** in the sidebar.
2. Select the saved JSON file under **Config file**.
3. Click **Load config**.

You can load a config after a restart or during a session.
The loaded config replaces the current workspace settings.
The application recalculates applied background fits from their saved parameters when it loads the spectra.
It keeps settings for both visualization modes when you change between spectra and 2D scan maps.

Keep the original NumPy or Zarr data at the saved paths.
If the data moves, select its new folder.
Then select the scans again.
Labels and styles for each file use its original path.
The application reports missing scans.

## NumPy and Zarr input

In **Data folder**, select the folder that contains `.npy` files or Zarr array directories (`.zarr` or `.zarray`).
You can select both formats under **Files to compare**.
Paths can include quotes, spaces, or escaped underscores and spaces.

You can also enter a Zarr array directory directly.
For example:

```text
/scratch/project_465002371/zuxian/torched_TACAW/BTO_Ba-O_bussi/tacaw_results_stemeels/scan_gpu_tacaw.zarray
```

Local Zarr v2 and v3 arrays support the same real numeric 3D and 6D shapes as NumPy.
They also support spectra, diffraction previews, maps, background subtraction, and exports.
For a Zarr group, select the array directory inside the group.
The application does not support ZIP stores or remote URLs.

For an existing environment, run `.venv/bin/python -m pip install -r requirements.txt` once.
Then restart the application.
Use the application controls to set the time step, sampling stride, and energy order.
Zarr attributes do not replace these settings automatically.

### Memory use

NumPy reads data in small blocks.
Zarr reads only the chunks that contain the requested data.
It must decode each compressed chunk in full.

Each read operation has a cache for decoded data.
The cache keeps up to 64 MiB or one larger chunk.
The operation releases the cache when it ends.
Codec buffers use additional memory.

Large Zarr chunks can need more memory and time than NumPy reads.
The application shows a message for chunks above 256 MiB.
The example BTO array has uncompressed chunks of 3.54 GiB.
Use its `.npy` copy for faster access.
Alternatively, change the chunk size of a separate Zarr copy.

### Prepare Zarr spectra

For 6D Zarr scans, one probe selection also prepares spectra for adjacent probes in the same spatial chunk.
The cache keeps only the integrated spectra and normalization totals.
The operation releases decoded chunks after use.

Changes to normalization, energy calibration, or broadening use the cached spectra again.
Changes to detector geometry or source data require a new integration.

To prepare all probe positions:

1. Set the detector.
2. Click **Prepare all Zarr probe spectra**.

The application reads each selected 6D Zarr scan once and caches the reduced spectra.
This first read can take several minutes for large scans.
The application keeps prepared spectra in memory.
A restart clears the cache.

The preparation control appears if the selected scans fit within 256 spatial-tile cache entries.
For the BTO 2D scan, each tile contains 3 × 3 probes.
A total of 100 tiles contains all 900 probe positions.

Diffraction previews and angle-resolved maps continue to read raw detector data.
The prepared cache makes spectrum extraction faster when you add probe positions.

### Source data changes

File inspection reads metadata and checks file attributes without a read of array values.
Changes to metadata or nested chunk files invalidate cached results.

Use data from completed simulations.
A simulation can change data during a read.
That read can combine data from different simulation stages.
The application does not change input data.

## Run tests

Run these commands from the application directory:

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
```

Browser interaction tests require Playwright and a browser.
To use Firefox, run these commands:

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install firefox
EELS_BROWSER=firefox .venv/bin/python -m pytest -q tests/test_axis_browser.py tests/test_angle_browser.py tests/test_background_browser.py
```

To use an installed Chrome browser, run the complete test suite:

```bash
EELS_CHROME_PATH=/opt/google/chrome/chrome .venv/bin/python -m pytest -q
```

Use the actual path to your Chrome executable.
The complete suite includes detector, file order, and workspace browser tests.
Notebook reference tests skip if the optional `STEM-EELS.ipynb` file is absent.
Panel resize tests skip while adjustable panels are disabled.

## License

The project uses the MIT license.
See [LICENSE](LICENSE).
