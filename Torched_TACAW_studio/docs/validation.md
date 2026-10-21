# Validation record

Validated on 6 October 2026 in the requested workspace.
The interface uses its own `.venv`; the calculation smoke tests use the existing
`venvs/deepmd-ipi` Python and the supplied `torched-tacaw-main` engine.
EELS Studio, that simulation environment, and the engine source were left unchanged.

## Automated coverage

Final integrated run: **47 passed in 170.98 seconds**, with Firefox enabled and
all three external CPU calculation smoke tests enabled.

- Endpoint and native-grid coordinates: horizontal, vertical, diagonal, reversed,
  rectangular, periodic, and tilted scans.
- Configuration round trips, expanded engine YAML imports, all four supplied
  presets, invalid batch shapes, frequency bounds, and reciprocal-space ROIs.
- Trajectory discovery and exclusion rules, natural bead ordering, mixed i-PI
  position/cell units, extended XYZ, LAMMPS mappings, selections and stride,
  malformed input, atomic conversion failures, and inconsistent bead sets.
- Frozen run bundles, relative path resolution, shell quoting, container GPU
  environment forwarding, initialization ordering, local supervision, worker
  failures, cancellation, and preserving stores during resume.
- Chunk-wise NumPy copying against known arrays, completion guards, disk estimates,
  interrupted conversion, and refusing unrelated or changed existing output.
- Streamlit interaction tests for parameter edits, YAML validation, presets,
  run generation, and keeping trajectory selection after folder changes.
- A real Firefox browser check for arbitrary blank-space coordinate selection,
  atom snapping, and line/rectangle overlay updates.
- Actual CPU plane-wave, three-point line, and 2×2 calculations, using four-frame
  chunks and a 16×16 multislice grid. Each result is finite, its NumPy export equals
  its Zarr array, and a completed-run resume preserves its values.

The CPU test timeout includes environment startup. A cold shared-filesystem import
exceeded the original 180-second limit; the test now allows 360 seconds and cleans
up its worker process group on timeout.

Reproduce the complete suite with installed Playwright Firefox and a compatible
simulation runtime:

```bash
TACAW_BROWSER=firefox \
TACAW_SMOKE_PYTHON=/absolute/path/to/simulation/python \
TACAW_SMOKE_ENGINE=/absolute/path/to/torched-tacaw-main \
.venv/bin/python -m pytest -q
```

Use `PLAYWRIGHT_BROWSERS_PATH` if the browser was installed in a custom location.
The runtime tests skip without their two environment variables; browser tests skip
when a browser executable is unavailable.

## Resource-display update

The resource display was checked separately after the initial implementation:
20 resource tests, 5 Streamlit tests, and 40 existing scientific/I/O/browser tests
passed. The new tests compare estimates with known arrays and all four provided
engine configurations, check batch-dependent memory, timestep/stride FFT bins,
CPU versus GPU planning, Slurm worker counts, and file-size measurement.
A real check with the existing calculation Python also confirmed physical node RAM
reporting. GPU capacity reporting has mocked coverage; actual GPU measurements
still require a target allocation. No production calculations were started.

## Single Python editor update

Calculation now uses one locally served Python-highlighted Config editor instead
of grouped controls and an advanced YAML input. Verification includes 21 parser
cases (including all supplied presets), six Streamlit interaction tests, and
browser checks for syntax highlighting, applying values, preserving rectangle
endpoints, rejected arguments, reload, and plot-coordinate selection.
Unapplied drafts survive scan changes and prevent accidental run generation.
An actual tiny CPU plane-wave calculation used settings applied through this
editor, completed NumPy export, and resumed without changing its results.
Existing scientific/I/O/resource tests passed as well.

## Supplied production trajectories

Only metadata and selected frames were inspected. No production calculation or
full conversion of these large inputs was performed.

| Input | Frames | Atoms per frame | First-frame cell lengths (Å) |
| --- | ---: | ---: | --- |
| SiC | 4,001 | 17,280 | 30.159, 31.93555, 184.38 |
| Si Langevin | Text source; not counted in full | 7,280 | 27.15549, 26.88256, 199.69903 |
| BTO Ba–O | 81,001 | 25,000 | 40.23304, 40.14531, 200.51076 |
| STO Ti–O | 38,060 | 25,000 | 39.09047, 39.09516, 195.41456 |

## Checks still needed on destination machines

CUDA/ROCm execution, actual Slurm submission, multiple GPUs/nodes, and the current
LUMI/Dardel module and container installations need checks within allocations on
those machines. The generated scripts and state handling have automated coverage;
this workspace validation does not certify a remote machine's software stack.
Follow [the allocation smoke-test procedure](setup.md#cluster-smoke-tests) before
running a large calculation. CPU is the portable laptop default.
