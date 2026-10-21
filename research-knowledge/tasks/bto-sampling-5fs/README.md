# BTO production sampling: 5 fs

The user confirmed output every 5 fs, keeping the 1 fs integration timestep
and 10 ps equilibration + 10 ps production. The reference is the local
`2026_BTO/lmp_nve_langevin.in`: its timestep already matches, and its dump
frequency is five steps. Its longer, restart-dependent run target is not adopted.

Both the original BTO working scripts and the preserved experiment now default
to `dump_every = 5` in the LAMMPS input and Python runner. Frame times follow
from the actual saved step numbers, giving 0.005 ps spacing and 2001 frames for
10,000 production steps including step zero. Sparse equilibration snapshots
remain every 1000 steps. The reference input itself is unchanged.

See [the timing evidence](../../../research-workflow/docs/bto-sampling-5fs-2026-10-21.json).
Earlier 10 fs trajectories, run metadata, cluster evidence and fingerprints
remain historical. This change does not establish spectral convergence or
authorize a new cluster/production submission. Resume with:

```bash
./rw context --task bto-sampling-5fs --site arrhenius
```
