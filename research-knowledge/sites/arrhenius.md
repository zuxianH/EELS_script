# Arrhenius observations — BTO, 21 October 2026

Direct inspection found a Slurm CPU partition, x86-64 architecture, system
Python 3.9.25, and an available centrally provided LAMMPS module. The original
input referenced `/proj/...` DeePMD/model locations unavailable in this session.
No central DeepMD module was found by the targeted probe. These findings are
observations from this session, not a permanent inventory of all site software.

An isolated local Python 3.11.17 environment was installed beside the original
BTO working directory. Recorded packages include DeePMD 3.2.0, LAMMPS
2025.7.22.2.0, TensorFlow CPU 2.21.0, MPICH 5.0.2 and ASE 3.29.0; see the
experiment's exact runtime lock. The packaged LAMMPS launcher discovers its
matched DeePMD plugin. A CUDA runtime dependency and explicit CPU selection
were needed for the wheel's C++ backend on this host; inference does not use
an allocated GPU. Mixing it with another MPI installation was not tested.

Two bounded, manually submitted jobs were observed terminal via `sacct`:

| Job | Atoms | Equilibration + production steps | CPU threads | Hard walltime | Result |
| --- | --- | --- | --- | --- | --- |
| 4212001 | 40 | 200 + 200 | 4 | 10 min | COMPLETED, 0:0; elapsed 59 s |
| 4212065 | 25,000 | 10 + 20 | 16 | 10 min | COMPLETED, 0:0; elapsed 2 min 38 s |

Both used 1 fs integration and saved production every 10 steps. Full-size
production took 68.9807 s for 20 steps; reported batch MaxRSS was 12,461,960 KiB
(about 11.9 GiB). These are short-run measurements, not guaranteed throughput
or memory bounds. Private account identifiers and raw accounting/log output
remain in the original local evidence store.

No standing verification budget, active workflow release or automatic cluster
submission is configured. `arrhenius` is registered only for scoped context.
The separate 10 ps + 10 ps manual production template is not submitted or
validated by this record. LUMI/Dardel remain separate, unvalidated targets.
