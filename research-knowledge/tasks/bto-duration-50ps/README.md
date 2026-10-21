# BTO: 50 ps total dynamics

The user explicitly selected **10 ps equilibration + 40 ps production**, keeping
the 1 fs integration timestep and 5 fs production sampling. The initial
interpretation of 50 ps equilibration was clarified before any edits.

Both the local BTO scripts and the preserved experiment now use 10,000
equilibration steps and 40,000 production steps. The production trajectory
contains 8001 saved frames including its step-zero frame. Integration and
sampling frequencies, temperature, potential and minimization are unchanged.

Both Slurm scripts request 60 hours, 16 CPU threads and 40 GB RAM. At the
previously measured 3.449035 s/step, 50,000 MD steps suggest 47.9 hours before
variability and I/O. The observed CPU partition maximum is 72 hours. No job is
submitted by this update, and convergence/domain stability remain unestablished.

See [current configuration evidence](../../../research-workflow/docs/bto-duration-50ps-2026-10-21.json).
Earlier configuration manifests and raw validation outputs remain historical.
Resume with `./rw context --task bto-duration-50ps --site arrhenius`.
