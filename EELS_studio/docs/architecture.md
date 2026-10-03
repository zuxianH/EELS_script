# Architecture

`app.py` calls `eels_studio.ui.application.main()`. The application sets up the
existing page, selects and inspects scans, then routes to the spectrum workflow
or scan-map view. Launch commands, `run_app.sh`, requirements, and `.streamlit`
configuration keep their existing roles. The default data folder remains the
directory containing `app.py`.

| Location | Responsibility / common changes |
| --- | --- |
| `core/spectra.py` | Detector geometry, bounded spectrum/strip/map extraction, energy calibration, FFT ordering, Gaussian broadening, reduced-tile normalization |
| `core/maps.py` | Parse energy requests, select inclusive bins, and accumulate energy-broadened scan maps |
| `core/background.py` | Background models, fitting, diagnostics, and fitting-domain selection |
| `core/integration.py` | Signed peak areas, endpoint interpolation, and area normalization |
| `core/display.py`, `core/identity.py` | Shared numerical display transforms and stable curve identities |
| `io/scans.py`, `io/scan_reader.py`, `io/scan_sources.py` | Metadata, revision checks, discovery, bounded NumPy/Zarr reads |
| `io/*_exports.py` | Numerical CSV, NPY, NPZ schemas and metadata |
| `ui/application.py` | Page setup, file selection, labels, and visualization routing |
| `ui/spectra.py` | Spectrum/probe controls, detector preview, analysis tabs, and download controls |
| `ui/background_view.py`, `ui/angle_resolved.py`, `ui/scan_map_view.py` | Analysis-specific interfaces and figure exports |
| `ui/plotting.py` | Shared line styles, spectrum labels, and Matplotlib downloads; initializes pandas before Plotly validation |
| `ui/cache_layer.py` | Shared Streamlit caching for metadata, spectra, reduced Zarr tiles, diffraction planes, fits, and numerical exports |
| `ui/state.py`, `ui/workspace_config.py` | Background lifecycle/fingerprints and versioned workspace settings |
| `ui/components/` | Browser interaction wrappers beside their unchanged JavaScript resources |

All paths in the table are under `eels_studio/`. Each package directory has an
`__init__.py`; callers import the owning module directly. `core` and `io` do not
import Streamlit, session state, or `ui`. Extraction functions accept scan metadata
and use `io` readers; the map reducer accepts a one-energy-map reader so the UI
can supply its cached reader without pulling Streamlit into the calculation.

## Behavior to preserve

Spectrum processing remains detector integration with optional full-probe
normalization → FFT ordering → optional Gaussian broadening → optional E²
weighting/background fit → display. Keep array axis order, float64 accumulation,
meV/fs units, integer detector centers, inclusive masks/windows, reflecting
Gaussian boundaries, and 4σ truncation unchanged. Display transforms are shared
by interactive spectra and figure exports; numerical exports retain linear data.
Angle-resolved plotting likewise shares its transformed array and labels with
figure downloads.

NumPy reads remain bounded, rather than loading or mapping entire scans. Zarr's
per-operation decoded-chunk cache keeps up to 64 MiB or one larger chunk. The UI
caches reduced spatial tiles (up to 256 entries), not decoded source chunks.
Source revisions include metadata and nested chunk stats; detector geometry is
part of extraction cache keys. Background fingerprints exclude labels, styles,
and plot limits but include scientific inputs and source revisions. Keep cached
wrappers in `ui` and retain their current entry limits when changing views.

Widget keys, component registration names, callback event fields, workspace JSON
schema, and export field names are compatibility boundaries. Python component
wrappers resolve JavaScript relative to `__file__`; register components inside the
active app runtime. Analysis tabs still render eagerly, as before. Adjustable
panels remain disabled; saved panel settings remain compatible.

## Verification

Run tests from the directory containing `app.py` using the README commands.
Existing tests cover numerical fitting/extraction, array layouts and endian
variants, bounded reads, Zarr cache invalidation, workspace restoration, exports,
and browser gestures. Targeted package tests check that scientific and numerical
I/O imports do not load Streamlit and that map reduction matches a full-array
convolution reference at energy boundaries under both FFT orderings. Optional
notebook comparisons and disabled panel gestures report skips explicitly.
