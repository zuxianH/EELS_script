# Torched TACAW Studio

Torched TACAW Studio is a browser interface for preparing and running vibrational EELS calculations.
It supports plane-wave illumination, STEM line scans, 2D scans, and synchronized bead trajectories.
It follows the Streamlit and Plotly interface used by EELS Studio.

## Start the interface

Use Python 3.11 or later.
Run these commands from this folder:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_app.sh
```

Open **http://localhost:8502**.
The server listens on the local computer only.
Use `./run_app.sh --server.port 8503` to select another port.
Use an SSH tunnel to access an interface running on a cluster.

The interface environment does not need Torch, pyms, or torched_tacaw.
Configure a separate calculation Python or container in **Machine & SBATCH**.
See [Setup on laptops and clusters](docs/setup.md) for the full installation procedure.
On this workspace, select the detected **Workspace CPU** profile to use the existing Torch/pyms environment and engine source without modifying them.

## Prepare a calculation

1. Enter the input folder in the sidebar, or use **Browse folders**.
2. In **Inputs**, select a trajectory. ASE `.traj` files appear first.
3. Enable **Synchronized bead set** to select multiple files for one calculation.
4. Set an explicit bead order if necessary. Numbered filenames sort numerically.
5. Select a calculation preset if useful. Presets change calculation settings and keep your selected inputs.
6. In **Scan**, select plane wave, line scan, 2D scan, unit-cell scan, or basis vectors.
7. In **Calculation**, edit the Python `Config(...)` block and click **Apply configuration**.
8. In **Machine & SBATCH**, configure the calculation environment and resource request.
9. In **Run & logs**, inspect the launch script and click **Generate run folder**.
10. Click **Check environment**, then **Run / submit** after the check finishes.

**Check environment** submits a short job for a Slurm profile.
It uses your current saved allocation settings.
GPU checks therefore run inside an allocation, rather than on the login node.
No calculation starts during folder generation.

## Select a scan on the trajectory

The Scan tab shows **Scan settings** on the left and the plot on the right.
Click-target and snapping controls sit above the plot.
Expand **Trajectory preview settings** or **Atom display settings** below the plot to edit those controls.
The plot shows one trajectory frame in the x–y plane.
Atom colors and the legend identify elements.
Hover shows the element, zero-based atom index, and coordinates.
Enable **Show element labels on atoms** for text on the plot.

Choose **Start / origin** or **End**, then click anywhere in the plot.
Enable snapping to select the nearest displayed atom.
You can also enter exact coordinates.
Drag to pan and scroll to zoom.
The overlay shows the actual probe coordinates used by TACAW.

A line scan includes the selected start and end.
A 2D scan uses the selected points as opposite rectangle corners and includes both corners.
For example, Start `(1, 1)` and End `(3, 4)` define a 2 Å wide, 3 Å high rectangle.
Both x and y coordinates must differ.
Switching from a line to 2D initializes the missing width/height and point count; edit these defaults to choose your region.
Each active axis needs at least two points.
The interface compensates for TACAW's native `i/N` grid convention.

Enable **Periodic sampling: exclude final edge** to scan a periodic region without duplicating the final boundary.
The BTO 2D preset uses this convention.
**Basis vectors** uses TACAW's raw origin and vectors without endpoint correction.
It allows tilted scan regions.
Imported `box` and `box_relative` scans retain their native convention and can be edited in the Python Config editor.

Use a z slab or **Display one z repeat** to reduce overlapping projected atoms.
Set the original repeat count explicitly.
These display filters do not remove atoms from the calculation.
Click **Load frame** after changing the frame index.
Negative frame indices work for indexed `.traj` files.
Text sources use nonnegative indices and may need to stream through earlier frames.

## Trajectory conversion

Existing ASE trajectories are referenced directly by default.
XYZ, extended XYZ, i-PI XYZ, and LAMMPS text dumps are converted to ASE trajectories before calculation.
Conversion reads one frame at a time.

All frames are retained unless you set start, stop, or stride.
The stop index is exclusive.
Converted files are stored inside the run folder.
Existing source files are never overwritten.

i-PI position and cell units are handled independently.
For example, the Si source uses Angstrom positions and a Bohr cell.
Metadata-free XYZ requires a cell and PBC override.
LAMMPS inputs may require a type-to-element mapping such as `{1: Ba, 2: Ti, 3: O}`.

Cell overrides use Angstrom.
Unit overrides describe the source numbers.
After conversion, the prepared snapshot interval is:

```text
source timestep × conversion stride
```

The FFT interval additionally includes TACAW's chunk step.
The current engine requires a fixed, diagonal, positive cell with periodic boundaries.
It does not implement general triclinic multislice.
The interface reports unsupported cells before calculation.

Bead files must have equal frame counts, identical atom ordering, and matching cells.
Preparation checks all selected calculation frames.
It also compares recorded time indices when present.
When time metadata is absent, you must ensure bead synchronization.

## Run folders and results

The default destination is:

```text
<input folder>/tacaw_runs/<name>_<timestamp>_<unique suffix>/
```

Each run contains:

| Artifact | Purpose |
| --- | --- |
| `studio.yaml`, `parameters.yaml`, `profile.yaml` | Frozen settings and user inputs |
| `run.py`, `tacaw_studio/` | Runner and application source snapshot |
| `run_local.sh`, `submit.sh`, `worker.sh` | Initial local and Slurm execution |
| `resume_local.sh`, `submit_resume.sh` | Continue unfinished work |
| `check.sh`, `submit_check.sh` | Check the configured runtime |
| `convert.sh`, `submit_convert.sh` | Convert a completed result |
| `trajectories/` | Converted inputs, when needed |
| `provenance.yaml`, `structure.extxyz` | Source records and first frame |
| `scan_coordinates.npy`, `scan_coordinates.csv` | Actual probe coordinates |
| `scan_gpu_config.yaml` | Engine configuration and mutable batch status |
| `scan_gpu_tacaw.zarray` | Intensity result, unless renamed |
| `job.json`, `lifecycle.json`, `environment.json`, `logs/` | Progress, runtime information, and logs |

The trajectory preparation and engine configuration files appear when execution starts.
Keep the referenced source trajectories and simulation environment available.
Run folders snapshot the runner code, but do not copy the engine or its dependencies.

The interface can reopen an existing run folder after restart.
Local jobs use a separate supervisor and continue when the browser disconnects.
Use **Refresh status and logs** for updated progress.
Use **Cancel job** to stop the local process group or request Slurm cancellation.

Resume resets unfinished batch statuses once before workers start.
It retains the existing Zarr array.
Do not edit input settings in a prepared run.
Generate a new run when changing parameters.
A failed partial preparation also needs a new run folder.

The engine owns batch writes and completion markers.
An interruption between a result write and its completion marker follows the engine's existing restart semantics.
For strict recovery after such an interruption, start a fresh run.

## Disk, RAM, and GPU estimates

**Run & logs** shows estimates for the current settings before generating a run,
and for the frozen settings of an opened run. Prepared runs use their saved engine
axes and array dimensions.

The summary reports uncompressed Zarr data, the extra NumPy file, approximate RAM
per worker, and approximate GPU VRAM per worker. CPU profiles use RAM for simulation
arrays and show no GPU requirement. Expand **Size and memory breakdown** for the
wave/FFT/multislice arrays, output chunk, RAM per node, converted positions, and
free space on the destination filesystem. GiB means 1024³ bytes. Filesystem free
space does not account for project quotas.

The memory model follows the supplied engine's complex128 exit-wave array,
float64 output, and pyms float32/complex64 multislice buffers. Its planning value
adds 50% headroom and 1 GiB per worker. This is an estimate, not a measured peak or
an upper bound; Torch, FFT workspaces, allocator caching, codecs, and other engine
versions can need more memory. The NumPy estimate includes a small header allowance.
Zarr compression determines its actual stored size. Converted trajectory estimates
cover positions only; text trajectories have an unknown frame count until conversion.

Reduce points per batch or frames per chunk to lower wave/FFT memory. Grid size,
ROI size, and multislice slices also affect working memory. Total scan points and
output/frequency ROIs affect the final file size. The engine averages trajectory
chunks into one result, so it does not store one complete output per trajectory chunk.

Use **Measure files on disk** to count existing run files without decoding arrays.
Repeat it to refresh the measurement while a job progresses. Externally referenced
source trajectories are excluded. **Check environment** records physical node RAM
and, for GPU profiles, the GPUs visible in that allocation and their total VRAM.
Physical node RAM may exceed the Slurm job's permitted memory.

## NumPy conversion and EELS Studio

Automatic NumPy conversion is disabled by default.
Enable it before generating the run, or use **Convert to NumPy** after calculation.
Conversion requires all batches to have status 2.
It decodes one Zarr chunk at a time and publishes only a complete NumPy file.
The interface displays estimated output and decoded chunk sizes.
Large chunks still require memory for the full decoded chunk and codec buffers.

Open the result Zarr directory or its NumPy copy in EELS Studio.
Use the timestep and chunk step stored in the engine configuration for energy calibration.
The output axis order remains `(dummy, energy, probe_x, probe_y, px, py)`.

## Save settings and profiles

**Download Studio settings** saves the input selection, scan, conversion, calculation, machine, and display settings.
Load it through **Studio or TACAW YAML**.
Existing TACAW configuration YAML can also be imported.
The interface extracts user inputs and recalculates derived values for new runs.
Missing trajectory paths must be corrected on the destination machine.

**Calculation** contains one Python-highlighted `Config(...)` editor.
It is pre-filled from the selected preset and current scan/settings.
Edit values and click **Apply configuration** in the editor, or press Ctrl+Enter
(Command+Enter on macOS). Valid changes update the same configuration used by
Scan, Inputs, Run & logs, and resource estimates. Invalid entries leave the last
applied settings intact. Apply draft edits before generating a run.

You can use concrete values, simple variables defined above `Config`, numeric
arithmetic such as `2/3`, `str(value)`, and simple f-strings:

```python
job_tag = "trial_01"
temperature = 300.0

config = Config(
    name=f"Si 110 {job_tag}",
    beam_energy_keV=60.0,
    beam_conv_ang_mrad=20.0,
    sample_temperature_K=temperature,
    sample_supercell_int=[1, 1, 1],
    kspace_bandwidth_limiting=[2/3, 2/3],
)
```

Arguments omitted from a pasted block keep their current applied values.
Replace command-line names such as `args.beam_energy_kev` with values or variables.
The editor parses settings without importing Torch or executing Python code.
Its highlighting assets are served locally through Streamlit using
[streamlit-code-editor](https://github.com/bouzidanas/streamlit-code-editor).

`trajectory_file` or `trajectory_files` changes the input selection.
Relative trajectory paths resolve under the current input folder.
`datafolder` sets the **parent** output folder; each Studio run still gets a new
unique subfolder. `intensities_zarray` sets the result basename inside that run.
`sample_structure_file=None` uses the first prepared trajectory frame.
`config_file`, trajectory-reader internals, derived axes, and batch status are
managed by the runner. Unsupported keys produce an error.

Changing calculation values preserves an existing endpoint-defined line/rectangle.
Changing native scan arguments in the code imports that geometry into Scan;
parallelograms use **Basis vectors** mode. Applied settings synchronize into the
editor when you change Scan or Inputs. An unapplied draft is retained, with a
notice if other settings changed. **Reload current settings** replaces that draft
with the current values. Download applied Python or TACAW YAML below the editor.
Existing TACAW YAML import remains available in the sidebar.
This adapter targets the supplied engine's 0.3.1 input interface.

Save named machine profiles on the current computer or download them.
Profiles contain Python/container commands, modules, dependency paths, environment variables, and worker binding.
Additional setup commands execute as shell commands in the chosen runtime.

## Tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Browser tests skip if the selected browser is unavailable.
Install Chromium with `.venv/bin/python -m playwright install chromium`, or set `TACAW_BROWSER_PATH` to an existing executable.
For Firefox, install it with `.venv/bin/python -m playwright install firefox` and run tests with `TACAW_BROWSER=firefox`.

See [Validation results](docs/validation.md), [Architecture](docs/architecture.md), and [Cluster smoke tests](docs/setup.md#cluster-smoke-tests).
