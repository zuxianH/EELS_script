# BTO: single-domain minimization and NVT

Scope: the supplied BTO compressed potential and bounded standalone-LAMMPS
checks on Arrhenius CPU, 21 October 2026. Evidence is indexed in
[`bto-single-domain-arrhenius-2026-10-21.json`](../../research-workflow/docs/bto-single-domain-arrhenius-2026-10-21.json).

The user needs a single-domain 300 K MD trajectory for subsequent energy-loss
spectra. The reported motivation is that previous large-supercell NPT produced
multiple domains. No trajectory from that earlier NPT run was supplied or
diagnosed; its cause is unknown. The requested alternative is small-cell energy
minimization, then replication and fixed-volume NVT.

## Definition and decisions

- Displacement means Ti relative to the centroid of its six surrounding O
  atoms, with periodic images handled locally. It is not a Ti–O bond length.
- **13 pm = 0.13 A**, not 1.3 A. Treat 13 pm as a proposed experimental
  comparator; its exact temperature, phase and measurement definition need
  confirmation. It is not an enforced equilibrium target.
- Single domain concerns spatial alignment of local polarization order.
  Displacement magnitude is a separate observable. Distinguish the mean local
  magnitude from the magnitude of the vector mean, since oppositely directed
  local vectors can cancel.
- The user explicitly retained the original `custom 3.97`, `a1 1.1` lattice
  and Ti fractional x=0.85. Its starting x length is 4.367 A and Ti displacement
  is +0.4367 A relative to the initial O cage.
- The implemented five-atom periodic minimization stays on the +x tetragonal
  branch with transverse coordinates restricted only during preparation.
  Coupled transverse box strains preserve equivalent y/z axes. NVT releases
  every atomic coordinate; no field or polarization restraint is applied.
- A 0 K minimized cell is not a 300 K zero-pressure equilibrium cell. Finite-
  temperature stress, thermal expansion and domain stability need evaluation.

## Verified, limited findings

| Quantity | Observed value |
| --- | --- |
| Minimized cell x/y/z | 4.09372607385 / 3.96454978744 / 3.96454978744 A |
| Ti minus six-O center | +0.171536284377 A along x |
| Physical maximum force component | 2.82779100136e-9 eV/A |
| Replicated 10 x 10 x 50 cell | 40.9372607385 x 39.6454978744 x 198.227489372 A |
| Model mapping and cutoff | Ba, Ti, O; 6 A |
| Engine/environment | LAMMPS 22 Jul 2025 Update 2; DeePMD 3.2.0; Python 3.11.17 |

Jobs 4212001 and 4212065 completed successfully with readable trajectories.
The 25,000-atom initial energy divided by 5000 agrees with the five-atom
minimized energy at relative tolerance 1e-10. Six regression tests cover
periodic O images, boundary crossings, opposite-vector cancellation, zero
displacement, atom ordering and export units/origin. Direct ASE export agrees
with the existing EELS XYZ converter for all 21 small-test production frames.

Neither the 0.2 ps small-test production window nor the 0.02 ps full-size
production window establishes thermal equilibrium or long-time domain order.
The full-size check included 5000 Ti sites and only three production frames.

## Next evidence

After an explicitly authorized production run, assess temperature/energy/stress
stationarity, spatial time-averaged Ti vectors and displacement distributions
over independent time windows. If domains emerge, preserve the result and
investigate the model, volume, temperature and sampling; changing constraints
changes the dynamics and must be treated as a separate experiment. Establish
integration-step, saved-interval and window convergence for the intended
spectrum, and confirm the experimental reference definition.
