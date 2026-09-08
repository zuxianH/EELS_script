import pyms
from ase.io import Trajectory, read, write
from ase.geometry import find_mic
import sys
import numpy as np
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt
import logging
import os

# --- Input arguments ---
if len(sys.argv) != 8:
    raise SystemExit(
        "Usage: python tacaw_gpu.py iGPU nGPUs nCPUs beads chunk offset epsilon"
    )

iGPU = int(sys.argv[1])       # which GPU to use
nGPUs = int(sys.argv[2])      # total number of GPUs
nCPUs = int(sys.argv[3])      # total number of CPUs

beads = int(sys.argv[4])      # number of beads
chunk = int(sys.argv[5])      # chunk size
offset = int(sys.argv[6])     # offset for the chunks
epsilon = float(sys.argv[7])  # central finite-difference displacement scale

if not np.isfinite(epsilon) or epsilon <= 0.0:
    raise ValueError(f"epsilon must be finite and positive, got {epsilon!r}")

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
logging.info(f"Running on {iGPU=} out of {nGPUs=}, using {nCPUs=} CPUs")
logging.info(
    "Run parameters: beads=%d, chunk=%d, offset=%d, epsilon=%.12g",
    beads,
    chunk,
    offset,
    epsilon,
)

# --- Parameters ---
gridshape = [400, 399]        # grid shape in pixels
grid_bp = tuple([int(round(gridshape[i] * 0.6666666666666666)) for i in range(2)]) # grid shape in pixels for bandpass
eV = 60000                    # energy in eV
nslices = 4*52                # number of slices (multiple of number of supercell in z direction)
subslices = np.linspace(1.0/nslices, 1.0, nslices) # subslice thickness


tstep = 5           # time step in fs
stride = 3          # stride step
skip_therm = 500    # skip initial thermalization frames
reference_frame = skip_therm

# Cell changes are not included in this displacement derivative. These tolerances
# only allow insignificant floating-point differences in a nominally fixed cell.
cell_rtol = 1.0e-7
cell_atol = 1.0e-8

# Set True to perform one extra reference multislice calculation and report the
# even-in-epsilon finite-difference residual for the first processed snapshot.
debug_linear_response = False


def calculate_exit_wave(atoms_config, incoming_wave):
    """Run the unchanged full multislice calculation for one configuration."""
    natoms_config = len(atoms_config)
    atomlist = np.concatenate(
        [
            atoms_config.cell.scaled_positions(atoms_config.positions),
            atoms_config.numbers.reshape(natoms_config, 1),
        ],
        axis=1,
    )
    crystal_config = pyms.structure(
        atoms_config.cell.diagonal(),
        atomlist,
        np.zeros(natoms_config),
        np.ones(natoms_config),
    )
    P, T = pyms.multislice_precursor(
        crystal_config,
        gridshape,
        eV,
        subslices=subslices,
        nT=1,
        device=f"cuda:{iGPU}",
        # device="cpu"
        showProgress=False,
        displacements=False,
        fractional_occupancy=False,
        band_width_limiting=[2/3, 2/3],
    )
    exit_wave = np.asarray(
        pyms.multislice(
            incoming_wave,
            nslices,
            P,
            T,
            device_type=f"cuda:{iGPU}",
            # device_type="cpu"
            return_numpy=True,
            qspace_in=False,
            qspace_out=True,
            subslicing=True,
        )
    )

    if exit_wave.shape != grid_bp:
        raise ValueError(
            "Unexpected q-space exit-wave shape: "
            f"expected {grid_bp}, got {exit_wave.shape}"
        )
    return exit_wave


# Assumes all beads have the same number of frames.
traj_files = [Trajectory(f"simulation.pos_{k:01d}.traj", "r") for k in range(beads)]
tmax = len(traj_files[0])
# tmax = 1000
window = scipy.signal.windows.tukey(chunk, alpha=1.0, sym=False)

# This common, immutable structure defines the expansion point R_0 for every
# bead and every time frame. It is intentionally taken from bead 0 only.
atoms_ref = traj_files[0][reference_frame].copy()
ref_positions = atoms_ref.positions.copy()
ref_cell = atoms_ref.cell.copy()
ref_numbers = atoms_ref.numbers.copy()
ref_pbc = atoms_ref.pbc.copy()
logging.info(
    "Using bead 0, frame %d as the common reference structure R_0",
    reference_frame,
)


def displacement_from_reference(atoms_config, bead_index, snapshot_index):
    """Return R(t) - R_0 as minimum-image Cartesian displacement vectors."""
    current_cell = np.asarray(atoms_config.cell)
    reference_cell_array = np.asarray(ref_cell)
    if not np.allclose(
        current_cell,
        reference_cell_array,
        rtol=cell_rtol,
        atol=cell_atol,
    ):
        max_cell_difference = np.max(np.abs(current_cell - reference_cell_array))
        raise ValueError(
            "The simulation cell differs from the reference cell at "
            f"bead {bead_index}, snapshot {snapshot_index} "
            f"(maximum absolute difference {max_cell_difference:.6g} Angstrom). "
            "Linearizing atomic displacements for a changing NPT cell requires "
            "separate strain/cell treatment."
        )
    if not np.array_equal(atoms_config.pbc, ref_pbc):
        raise ValueError(
            "Periodic boundary conditions differ from the reference at "
            f"bead {bead_index}, snapshot {snapshot_index}."
        )
    if len(atoms_config) != len(ref_numbers) or not np.array_equal(
        atoms_config.numbers, ref_numbers
    ):
        raise ValueError(
            "Atom count, ordering, or chemical species differ from the common "
            f"reference at bead {bead_index}, snapshot {snapshot_index}."
        )

    raw_displacement = atoms_config.positions - ref_positions
    displacement, _ = find_mic(
        raw_displacement,
        cell=atoms_config.cell,
        pbc=atoms_config.pbc,
    )
    return displacement


# Use the reference structure to generate the incoming wavefunction. The same
# incoming wave is passed to both finite-difference multislice calculations.
natoms = len(atoms_ref)
reference_atomlist = np.concatenate(
    [
        atoms_ref.cell.scaled_positions(atoms_ref.positions),
        ref_numbers.reshape(natoms, 1),
    ],
    axis=1,
)
reference_crystal = pyms.structure(
    ref_cell.diagonal(),
    reference_atomlist,
    np.zeros(natoms),
    np.ones(natoms),
)
psi = pyms.plane_wave_illumination(gridshape, reference_crystal.unitcell[:2], eV)

# Final intensity accumulator
Iqo = np.zeros((chunk, *grid_bp), dtype=float)
nchunks = 0
debug_check_done = False

# --- Loop over chunks (time) ---
for ichunk in tqdm(np.arange(skip_therm, tmax - chunk * stride, offset)):
    avg_psit = np.zeros((chunk, *grid_bp), dtype=np.complex128)

    # --- Loop over beads ---
    for k in range(beads):
        traj = traj_files[k]
        psit_chunk = np.empty((chunk, *grid_bp), dtype=np.complex128)

        for i in range(chunk):
            isnap = ichunk + i * stride
            atoms = traj[isnap]
            displacement = displacement_from_reference(atoms, k, isnap)

            # Both configurations are constructed about the same R_0. Do not
            # perturb the instantaneous configuration or mutate trajectory data.
            atoms_plus = atoms_ref.copy()
            atoms_minus = atoms_ref.copy()
            atoms_plus.positions = ref_positions + epsilon * displacement
            atoms_minus.positions = ref_positions - epsilon * displacement
            atoms_plus.wrap()
            atoms_minus.wrap()

            # Full dynamical multislice propagation is retained on both sides of
            # the finite difference; only its displacement dependence is linearized.
            psi_plus = calculate_exit_wave(atoms_plus, psi)
            psi_minus = calculate_exit_wave(atoms_minus, psi)
            psi_linear = (psi_plus - psi_minus) / (2.0 * epsilon)

            if psi_linear.shape != grid_bp:
                raise ValueError(
                    "Unexpected linearized exit-wave shape: "
                    f"expected {grid_bp}, got {psi_linear.shape}"
                )
            psit_chunk[i, ...] = psi_linear

            if debug_linear_response and not debug_check_done:
                atoms_zero = atoms_ref.copy()
                atoms_zero.wrap()
                psi_zero = calculate_exit_wave(atoms_zero, psi)
                even_residual = (psi_plus + psi_minus) / 2.0 - psi_zero
                zero_norm = np.linalg.norm(psi_zero)
                relative_error = (
                    np.linalg.norm(even_residual) / zero_norm
                    if zero_norm > 0.0
                    else np.linalg.norm(even_residual)
                )
                logging.info(
                    "Linear-response self-check at bead %d, snapshot %d: "
                    "||(psi_plus + psi_minus)/2 - psi_zero|| / ||psi_zero|| "
                    "= %.6e (epsilon=%.12g)",
                    k,
                    isnap,
                    relative_error,
                    epsilon,
                )
                debug_check_done = True

        # Apply the same Tukey window and coherently accumulate amplitudes.
        psit_chunk = np.einsum("i,ijk->ijk", window, psit_chunk)
        avg_psit += psit_chunk

    # Average wavefunction amplitudes across beads before the time FFT/intensity.
    avg_psit /= beads

    # FFT the complex, bead-averaged linear exit wave and accumulate |FFT|^2.
    psio_chunk = scipy.fft.fft(avg_psit, axis=0, workers=nCPUs)
    Iqo += np.real(psio_chunk * psio_chunk.conjugate())
    nchunks += 1

if nchunks == 0:
    raise RuntimeError(
        "No time chunks were processed. Check tmax, skip_therm, chunk, stride, "
        "and offset."
    )

# Preserve the original per-chunk and per-time-sample normalization. There is no
# epsilon^2 factor because psi_linear already estimates dPsi/dlambda at lambda=0.
Iqo = np.fft.fftshift(Iqo, axes=(0, 1, 2)) / nchunks / chunk

# Save final results. Avoid a period in the epsilon part of the filename.
epsilon_tag = np.format_float_positional(epsilon, trim="-").replace(".", "p")
output_filename = (
    f"Iqo_si110_linear1ph_bead{beads}_chunk{chunk}_offset{offset}_"
    f"eps{epsilon_tag}.npy"
)
np.save(output_filename, Iqo)
logging.info(
    "Finished processing all chunks and beads with epsilon=%.12g; saved %s",
    epsilon,
    output_filename,
)
exit()
