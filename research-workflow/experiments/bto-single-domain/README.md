# BTO single-domain preparation

Prepare one periodic five-atom cell, minimize its +x tetragonal branch, replicate
it, then equilibrate and sample with unrestricted 300 K NVT. This preserves the
user's original starting structure: `custom 3.97`, `a1 1.1`, and Ti basis
`0.85 0.75 0.75`. The local potential is supplied separately.

The [Arrhenius evidence](../../docs/bto-single-domain-arrhenius-2026-10-21.json)
records the tested original scripts and model hash. This directory preserves
their scientific inputs and analysis code. The environment-location defaults
and submission template were adapted for portability and checked locally;
the portable Slurm template has not been cluster submitted.

The later [5 fs sampling update](../../docs/bto-sampling-5fs-2026-10-21.json)
changes production output frequency while retaining 1 fs integration and
10 ps equilibration + 10 ps production. The original evidence manifest refers
to the earlier snapshot and its 10 fs test trajectories; use the update's
fingerprints for the current sampling defaults.

## Environment and local evidence

`BTO_WORKDIR` denotes the original local `LAMMPS/BTO` working directory,
outside this Git repository. Set it explicitly on the current machine. Model,
raw logs, Slurm outputs, restart/data files and trajectories remain there.
The published records use this alias rather than personal filesystem paths.

On the existing Arrhenius installation, reuse its environment:

```bash
export BTO_SOFTWARE_ROOT="$BTO_WORKDIR/../.software"
source env.sh
```

Or run `bash install_deepmd.sh` to create a new isolated `.software/` directory
here. `requirements-runtime.lock` pins the installed package versions; a clean
reinstallation from this lock has not yet been exercised. This is an x86-64 CPU
execution environment. The packaged DeePMD plugin includes CUDA components,
so the small runtime dependency is installed and `CUDA_VISIBLE_DEVICES=-1`
selects CPU operation. No private key or system installation is included.

## Scientific settings

- `lmp_minimize.in`: relax all species along x in a five-atom periodic cell.
  `fix setforce NULL 0 0` preserves the tetragonal branch during preparation.
  Default box relaxation targets zero external stress, coupling y/z strains.
  `--relax-box 0` instead retains the original dimensions. This is a
  symmetry-restricted 0 K preparation, not a 300 K zero-pressure cell.
- `lmp_nvt.in`: read the minimized cell; replicate 10 x 10 x 50 (25,000 atoms);
  assign 300 K velocities; run NVT with every coordinate free. The default
  integration step is 1 fs, equilibration 10 ps, production 10 ps, saved interval
  5 fs (every five steps; 2001 production frames including step zero).
  Sparse equilibration snapshots remain every 1 ps. Durations are provisional
  and need convergence checks.
- `run_workflow.py`: record model/runtime metadata, refuse existing output,
  check the physical final force and +x seed displacement, and run domain
  analysis. Physics stays in these experiment inputs, outside site profiles.
- `trajectory_tools.py`: calculate Ti minus the centroid of six nearest
  periodic oxygen images; write per-frame and per-Ti time averages. Export to
  ASE `.traj` or extended XYZ with species, stable ordering, box and timing.
  Primitive cells require distinct periodic images of the same O atom IDs.

The cell minimization excludes spatial domain walls by construction. NVT
stability remains a measurement: all coordinates are released. No field or
polarization restraint is imposed. Inspect spatial, time-averaged Ti vectors;
neither a global displacement mean nor instantaneous thermal sign changes
alone establish the domain pattern. A seventh-neighbor/topology analysis may
be needed if the octahedra cease to be well separated.

## Manual execution

The potential is not committed. Its tested SHA-256 is
`9d9c070bc34b362c4bdb71bc8090300ed59abccbe88bf7108d7119fce8a422d6`.
After setting `BTO_WORKDIR` and activating the environment:

```bash
python run_workflow.py --help
python -m unittest -v test_trajectory_tools.py
```

Submit from this experiment directory with your explicitly chosen CPU account:

```bash
sbatch --account=YOUR_CPU_ACCOUNT submit_arrhenius.slurm \
  --model "$BTO_WORKDIR/BTO_pbesol_final_compressed.pb" \
  --output results/single_domain_300K
```

The template requests one task, 16 CPU threads, 40 GB RAM and 36 hours on `cpu`.
The original 25,000-atom check measured 3.449 s/step; the default 20,000 MD
steps therefore suggest about 19–20 hours before variability and I/O. This
estimate does not validate the portable template or scientific convergence.
Submission requires a separate explicit production request; these records do
not authorize execution. `rw` itself has no submission capability.

For minimization only, add `--stage minimize`. To reuse a minimized cell, use
`--stage nvt --cell PATH_TO_FIVE_ATOM_DATA`. Inspect domain CSV/JSON outputs
before spectra. Convert only a uniformly sampled production window:

```bash
python trajectory_tools.py convert results/single_domain_300K/nvt/production.lammpstrj \
  results/single_domain_300K/production.traj --dt-ps 0.001
```

Downstream spectra must use the saved frame interval, default 5 fs, rather
than the 1 fs integration step. Actual exported XYZ was also checked against
the existing `EELS_script/trajectories/convert_xyz_traj.py` reader.
That original converter check used 10 fs test output. Existing trajectories
retain their own timing; changing defaults does not change those files.

## Evidence limits

Original Slurm jobs 4212001 (40 atoms, 200+200 steps) and 4212065 (25,000 atoms,
10+20 steps) completed with `0:0`; the tests used 1 fs integration. Six physical
and conversion regression tests passed. The relaxed seed is approximately
4.093726 x 3.964550 x 3.964550 A, with Ti-to-six-O displacement +0.171536 A.

These are short execution checks. They do not establish 300 K equilibration,
long-time single-domain stability, agreement with a 13 pm experimental value,
spectral convergence, i-PI shutdown behavior, or a production release. See the
[task handover](../../../research-knowledge/tasks/bto-single-domain/README.md).
