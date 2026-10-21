# BTO: 415 ps restartable NVT

The user superseded the earlier 50 ps total target and authorized the longer
calculation: 10 ps equilibration plus 405 ps production, 1 fs integration and
5 fs production sampling, giving 81,001 unique frames. Nine dependent segments
are planned with 16 CPU threads, 40 GB RAM and 60 hours per job.

The new segment runner freezes input/model/seed, keeps a continuous clock,
restores the thermostat and atom state, saves 1 ps checkpoints and records
trajectory ranges excluding shared boundaries and discarded tails. The old
50 ps one-job entry point remains a separate historical configuration.

The seed is the validated five-atom minimized +x tetragonal cell. No added
polarization restraint is used in NVT. Finite-temperature domain stability
remains a scientific question. See the [restart workflow](../../../research-workflow/experiments/bto-single-domain/README_segmented.md)
and the task's append-only evidence for checks, source hashes and job IDs.

Submitted production jobs **4215841–4215849**, with absolute endpoints 50000,
100000, 150000, 200000, 250000, 300000, 350000, 400000 and 415000. Each uses
`afterok` on its predecessor. At submission the first job was queued; all
successors were waiting on dependencies. Frozen code revision:
`2a569d5ed025bcb649abf7c95911820d1f2d7952`. Results: `BTO_WORKDIR/results/single_domain_300K_415ps`.
See [submission evidence](../../../research-workflow/docs/bto-long-nvt-submission-2026-10-21.json).
