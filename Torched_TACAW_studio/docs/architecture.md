# Architecture

`app.py` calls `tacaw_studio.ui.main()`.
The UI uses Streamlit and Plotly.
The runtime imports Torch, pyms, and torched_tacaw only in calculation commands.

| Module | Responsibility |
| --- | --- |
| `python_config.py` | Config-style Python rendering, bounded AST value parsing, and Studio/scan mappings |
| `config.py` | Versioned Studio settings, flat engine inputs, validation, preset and YAML adapters |
| `geometry.py` | Endpoint correction and native scan coordinates |
| `trajectories.py` | Discovery, single-frame previews, bounded readers, atomic conversion, bead checks |
| `profiles.py` | Machine profiles, Slurm resource validation, shell script generation |
| `jobs.py` | Frozen run bundles, local supervision, scheduler status, submission, cancellation |
| `runner.py` | Environment checks, preparation, worker execution, resume, completion |
| `resources.py` | Torch-free result, working-memory, destination-space estimates and explicit file-size measurement |
| `storage.py` | Atomic metadata and chunk-wise Zarr-to-NumPy conversion |
| `ui.py`, `scan_click.js` | Browser workflows and arbitrary-coordinate selection |

Scientific and I/O modules do not import Streamlit.
The UI does not import Torch or the TACAW engine.

## Interfaces

Studio settings use `schema_version: 1`.
The top-level sections are `inputs`, `parameters`, `scan`, and `execution`.
`parameters` contains flat user arguments for engine `Config`.
`scan` owns geometry, except imported native box geometry.
`effective_parameters()` converts the selected geometry into engine arguments.
The Python editor merges validated named Config inputs, preserves an unchanged endpoint scan, and imports explicitly changed native geometry.
Its AST parser handles literals, simple assignments/arithmetic, and f-strings without executing Python.
The locally served Ace component submits the current buffer atomically; draft and applied settings are tracked separately.

The runtime reads `studio.yaml`.
Its CLI actions are `check`, `prepare`, `resume`, `worker`, `finish`, and `convert`.
Calculation preparation generates the expanded engine configuration.
The engine remains responsible for derived physics and batch status updates.

Each run snapshots the package and launchers.
Source trajectories and simulation dependencies remain external references.
Settings in a generated run are frozen.
The UI reads engine batch statuses without changing them.

## Execution ordering

Initial launch performs environment check → trajectory preparation → one-time Config/Zarr creation → workers → completion check → optional NumPy conversion.

Slurm uses one shared preparation process followed by `srun`.
Local execution uses a separate Python supervisor.
The supervisor persists exit status.
Linux PID start-time identity prevents signalling a reused PID.
macOS uses process identity from the supervisor command.
Cancellation signals the entire local process group.

Resume verifies the result shape and source file revisions.
It resets status 1 to 0 under the engine's shared config lock.
It does not call `Master.restart_unfinished()`, because that method also immediately runs a worker.
Resume launch starts workers only after the one-time reset finishes.

## Browser component

The local JavaScript component reads Plotly axis transforms after an un-dragged primary click.
It converts browser coordinates to x/y coordinates.
It sends the coordinate, selection role, event ID, and preview ID to Streamlit.
The callback rejects stale previews and optionally snaps to displayed atoms.
It updates widget state before the next script rerun.
Pan and zoom remain native Plotly interactions.

## Extending the engine adapter

Update the supported input registry, validation, and YAML import mapping together when adding engine inputs.
Do not expose derived axes or mutable batch status as editable inputs.
Use `Config` to regenerate them.
Keep output shapes and `ij` scan axis order compatible with EELS Studio.

GPU support is currently CPU, NVIDIA CUDA, and AMD ROCm.
ROCm uses Torch's `cuda:N` device names.
Apple MPS is not an assumed pyms/TACAW backend.
Use CPU on an Apple laptop unless that backend is separately validated.
