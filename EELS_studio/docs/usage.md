# Analysis workflow

Run `./run_app.sh` from `EELS_studio`, then open http://localhost:8501.
Installation and test commands are in the [README](../README.md).

1. Choose a folder containing NumPy or local Zarr arrays. Select scans, arrange
   their comparison order, and optionally set labels. Arrays use
   `(energy, px, py)` or `(dummy, energy, probe_x, probe_y, px, py)`.
2. In **Spectra & detector**, choose probe positions and verify the time step,
   sampling stride, and input FFT ordering. Set `px`, `py`, and `radius` below
   the detector preview, or click the image to move the center.
3. Choose full-probe normalization and optional Gaussian broadening. Normalization
   uses all energies and detector pixels at each probe; broadening follows
   extraction and FFT ordering, using reflecting boundaries and a 4σ kernel.
4. Inspect spectra, adjust display limits/styles, and use **Background** to preview
   a fit before applying it. Peak areas integrate signed linear residuals with
   interpolated endpoints. **Angle-resolved EELS** uses an independent rectangular
   strip and reports detector pixels, without assuming angular calibration.
5. For spatial maps, choose **2D scan map**, select a 6D scan, and request energies
   or ranges such as `20, 40, 10-20`. Map controls are independent; these maps use
   raw detector sums without spectrum normalization.
6. Export numerical data for the full recorded range and figures using the explicit
   display controls. Browser zoom and hidden legend entries do not change exports.
   Background exports preserve masks and signed residuals. E²-weighted fits remain
   weighted in exports; the display never weights them twice.
7. Use **Workspace config → Save config** to save paths and analysis settings.
   Load that JSON to resume; applied fits are recalculated. Keep source data at the
   saved paths. Configs contain settings, not source arrays or prepared caches.

Energy is in meV; simulation time steps are in fs. Input data is read without
modification. NumPy uses bounded reads. Zarr decodes intersecting chunks in full,
so chunk size determines peak memory. **Prepare all Zarr probe spectra** caches
only reduced spectra for the current detector in the running application.
