# Phase A inventory — 21 October 2026

Baseline: EELS_script commit `e7a73d2`; initial working tree clean. Inventory
covered the repository, directory names under Documents, configured SSH aliases,
and noninteractive connection probes. No cluster files or jobs were changed.

## Observations

| Item | LUMI | Dardel |
| --- | --- | --- |
| Configured SSH alias | `lumi` | `dardel` |
| Noninteractive SSH result | public-key authentication rejected | public-key/GSSAPI authentication rejected |
| Live scripts, environment, process tree, logs and scheduler state | inaccessible | inaccessible |
| Existing TACAW template | `Torched_TACAW_studio/tacaw_studio/profiles.py` | same file |
| Template modules | LUMI/25.03; partition/G; PyTorch/20240918-rocm-6.2.1-python-3.12-singularity-20241007 | PDC; singularity/4.2.0-cpeGNU-24.11 |
| Template container | `${SIFPYTORCH}` with `--rocm` | PDC ROCm 6.3/Python 3.12/PyTorch 2.4 container, `/cfs/klemming` bind |

These are source templates, not observed installed environments or working
i-PI/LAMMPS launches. Module/container and filesystem differences are intentional
site choices in those templates; their current validity has not been tested.
There is insufficient evidence to identify accidental remote divergence.

The existing Studio snapshots runtime code into run folders, supervises local
TACAW processes, and generates Slurm scripts with `srun --kill-on-bad-exit=1`.
This is TACAW execution logic, not an i-PI/LAMMPS server/client supervisor.
`notebooks/inspect_ipi_output.ipynb` contains analysis/monitoring, not the real
MD launch scripts needed for the reported allocation issue. No `.slurm`,
`.sbatch`, or `input.xml` launch inputs were found by the bounded Documents scan.

Local tools observed: Python 3.12.3 and Codex CLI 0.160.0. `claude`, `pytest`, and
`sbatch` were not found on PATH. No software was installed into an existing
scientific environment. Git HTTPS reads work; terminal pushes previously failed
because no noninteractive GitHub credentials were available.

## Sync boundaries

Allow: reviewed source, small configuration/templates, tests, documentation,
sanitized append-only events and handovers. Review notebooks and archives
separately because embedded outputs may be large or restricted. Existing
tracked files remain intact; ignore rules do not remove already tracked data.

Exclude: credentials/private keys, `.env` files, environments, trajectories,
result arrays/stores, raw logs, checkpoints, run bundles and local workflow state.
No untracked research data was moved, deleted, or published by inventory.

## Unresolved regression and next evidence

The MD-finished/allocation-idle symptom is user-reported, not reproduced here.
Its cause remains unknown. Obtain authenticated access or sanitized copies of
both real launch scripts, environment setup, tiny inputs, server/client logs,
`squeue`/`scontrol` observations and `sacct` batch/step exit records. Distinguish
application RUNNING from scheduler COMPLETING. No job IDs were available.

Before any scheduler test, record site account/partition, executable and build,
nodes/CPUs/memory, hard wall-time, queue timeout, execution timeout, shutdown
grace, per-candidate attempts and daily compute budget. Leave them unset until
confirmed; do not infer them from production scripts. At most one verification
allocation per site is the proposed initial concurrency limit.

Phase B remains pending. Local worktree/knowledge tooling can be prepared
independently without changing existing simulation execution logic. Neither a
new release nor an automatic coordinator is activated by this bootstrap.
