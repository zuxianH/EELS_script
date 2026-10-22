# Restartable 415 ps BTO NVT

The current user choice is **one 60-hour job targeting all 415 ps**, followed
by manual continuation when needed. Prepare with `--segment-steps 415000` and
submit only endpoint `415000`. Resubmit that same endpoint after a timeout;
there is no automatic successor. The earlier nine-job plan below is an
optional historical arrangement, cancelled for the current run.

The selected run is **10 ps equilibration + 405 ps production** at 300 K,
with 1 fs integration and 5 fs production output: **81,001 unique frames**.
It reuses the validated five-atom minimized +x seed and BTO PBESOL potential,
replicated 10 x 10 x 50 (25,000 atoms). All atomic coordinates move freely.
The legacy `run_workflow.py` remains a separate 50 ps one-job entry point.

The continuous absolute clock is 0..415000. Production is 10000..415000;
elapsed production time is `(step - 10000)*0.001 ps`. Matching the reference
frame count does not make its potential, thermostat or preparation identical.

## Prepare and submit

Source the installed CPU environment, then prepare an empty output directory:

```bash
python segmented_nvt.py prepare --output results/single_domain_300K_415ps \
  --model BTO_pbesol_final_compressed.pb \
  --cell validation/minimization_cpu/minimized_cell.data
```

This copies the seed, potential and segment sources and records SHA256 digests,
runtime location/version and settings in `run_manifest.json`. Retain the seed's
minimization provenance. Commit source before submitting the frozen snapshot.

Each job requests one task, 16 CPU threads, 40 GB RAM and 60 hours. The estimated
total is 16.6 compute days, excluding queues, from a short benchmark. Runtime
is not guaranteed. The observed partition maximum is 72 hours. Endpoints are:

`50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 415000`.

Submit `RUN/source/submit_segment.slurm` with explicit `--account`, absolute
`--chdir=RUN`, per-job `--output`, and positional arguments `RUN ENDPOINT`.
Successors use `--dependency=afterok:JOBID`. The first job includes 10 ps
equilibration and 40 ps production; seven contribute 50 ps each; the last
contributes 15 ps. The runner itself never submits jobs or monitors the queue.

## Restart and output

Checkpoints `checkpoints/checkpoint.STEP` are saved every 1000 steps (1 ps),
including equilibration, and at successful segment ends. Continuation restores
atom positions, velocities and the Nose-Hoover state using the same thermostat
ID/style. It does not replicate the seed, regenerate velocities, repeat
equilibration or reset the clock. A lock rejects concurrent runs against one
output directory. Completed segments are idempotent.

After a timeout, resubmit the frozen script with the **same endpoint**. The
runner selects the highest numbered nonempty checkpoint. A truncated nonempty
checkpoint causes an explicit LAMMPS error: preserve it for diagnosis and move
it out of `checkpoints/` before trying an earlier valid checkpoint. Successors
with `afterok` will not run after a failure; cancel only the owned blocked jobs
and rebuild their dependency chain on the retry. There is no unbounded retry.

Each attempt has a separate directory under `attempts/`, preserving old output.
`progress.json` marks the last completed segment. `trajectory_index.json`
records accepted production ranges: the latest attempt owns a shared restart
boundary, and older overlaps or rolled-back tails are excluded.

After completion, assemble a uniformly sampled dump:

```bash
python RUN/source/segmented_nvt.py assemble --output RUN \
  --destination RUN/production.lammpstrj
```

Assembly streams data, requires every expected frame once, and refuses an
existing destination. On error it leaves an incomplete destination for
inspection; remove it explicitly before retrying. It needs another full
uncompressed copy and is not automatically run in the simulation job. Do not
concatenate raw segment files directly because boundaries overlap. Existing
analysis/export tools can read the assembled dump; its production origin is
recorded in the manifest/index.

Inspect temperature, energy, stress and spatial time-averaged Ti-to-six-O-center
vectors before spectra. Short restart checks establish neither equilibration
nor long-time single-domain stability.

References: [read_restart](https://docs.lammps.org/read_restart.html),
[run upto](https://docs.lammps.org/run.html),
[Nose-Hoover restart state](https://docs.lammps.org/fix_nh.html).
