"""Run the plane-wave GPU TACAW calculation on ASE bead trajectories.

Run from the repository root; use --input-dir to select the trajectory folder.
The six original positional arguments and calculation parameters are retained.
"""

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iGPU", type=int, help="GPU device index")
    parser.add_argument("nGPUs", type=int, help="total GPU count (logged only)")
    parser.add_argument("nCPUs", type=int, help="CPU workers for the FFT")
    parser.add_argument("beads", type=int, help="number of bead trajectory files")
    parser.add_argument("chunk", type=int, help="frames per FFT chunk")
    parser.add_argument("offset", type=int, help="offset between chunk starts")
    parser.add_argument("--input-dir", type=Path, default=Path.cwd(),
                        help="folder containing simulation.pos_N.traj (default: current directory)")
    parser.add_argument("--output", type=Path,
                        help="output .npy path (default: original filename inside input directory)")
    args = parser.parse_args(argv)
    args.input_dir = args.input_dir.expanduser().resolve()
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)

    import pyms
    from ase.io import Trajectory
    import numpy as np
    from tqdm import tqdm
    import scipy
    import logging

    iGPU, nGPUs, nCPUs = args.iGPU, args.nGPUs, args.nCPUs
    beads, chunk, offset = args.beads, args.chunk, args.offset

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info(f"Running on {iGPU=} out of {nGPUs=}, using {nCPUs=} CPUs")
    logging.info(f"Number of beads {beads=}, chunk {chunk=}, and offset {offset=}")

    # --- Parameters ---
    gridshape = [400, 399]        # grid shape in pixels
    grid_bp = tuple([int(round(gridshape[i] * 0.6666666666666666)) for i in range(2)]) # grid shape in pixels for bandpass
    eV = 60000                    # energy in eV
    nslices = 4*52                # number of slices (multiple of number of supercell in z direction)
    subslices = np.linspace(1.0/nslices, 1.0, nslices) # subslice thickness


    tstep = 5           # time step in fs
    stride = 3          # stride step
    skip_therm = 500    # skip initial thermalization frames
    # Assumes all beads have the same number of frames
    traj_files = [Trajectory(args.input_dir / f"simulation.pos_{k:01d}.traj", 'r') for k in range(beads)]
    tmax = len(traj_files[0])
    #tmax = 1000
    window = scipy.signal.windows.tukey(chunk, alpha=1.0, sym=False)

    # Use one bead to generate initial wavefunction
    atoms = traj_files[0][0]
    natoms = len(atoms)
    atomlist = np.concatenate([atoms.cell.scaled_positions(atoms.positions), atoms.numbers.reshape(natoms, 1)], axis=1)
    crystal = pyms.structure(atoms.cell.diagonal(), atomlist, np.zeros(natoms), np.ones(natoms))
    psi = pyms.plane_wave_illumination(gridshape, crystal.unitcell[:2], eV)

    # Final intensity accumulator
    Iqo = np.zeros((chunk, *grid_bp), dtype=float)
    nchunks = 0

    # --- Loop over chunks (time) ---
    for ichunk in tqdm(np.arange(skip_therm, tmax - chunk * stride, offset)):
        avg_psit = np.zeros((chunk, *grid_bp), dtype=complex)

        # --- Loop over beads ---
        for k in range(beads):
            traj = traj_files[k]
            psit_chunk = np.empty((chunk, *grid_bp), dtype=complex)

            for i in range(chunk):
                isnap = ichunk + i * stride
                atoms = traj[isnap]
                atoms.wrap()
                natoms = len(atoms)
                atomlist = np.concatenate([atoms.cell.scaled_positions(atoms.positions), atoms.numbers.reshape(natoms, 1)], axis=1)
                crystal = pyms.structure(atoms.cell.diagonal(), atomlist, np.zeros(natoms), np.ones(natoms))
                P, T = pyms.multislice_precursor(
                    crystal,
                    gridshape,
                    eV,
                    subslices=subslices,
                    nT=1,
                    device=f'cuda:{iGPU}',
                    #device="cpu"
                    showProgress=False,
                    displacements=False,
                    fractional_occupancy=False,
                    band_width_limiting=[2/3, 2/3]
                )
                psit_chunk[i, ...] = pyms.multislice(
                    psi,
                    nslices,
                    P,
                    T,
                    device_type=f'cuda:{iGPU}',
                    #device_type="cpu"
                    return_numpy=True,
                    qspace_in=False,
                    qspace_out=True,
                    subslicing=True
                )

            # Apply window and accumulate for this bead
            psit_chunk = np.einsum('i,ijk->ijk',window,psit_chunk)
            avg_psit += psit_chunk

        # Average across beads
        avg_psit /= beads

        # FFT and accumulate intensity
        psio_chunk = scipy.fft.fft(avg_psit, axis=0, workers=nCPUs)
        Iqo += np.real(psio_chunk * psio_chunk.conjugate())
        nchunks += 1

    # Normalize
    Iqo = np.fft.fftshift(Iqo, axes=(0, 1, 2)) / nchunks / chunk

    # Save final results
    destination = args.output if args.output is not None else (
        args.input_dir / f"Iqo_si110_avg_temp1000_bead{beads}_chunk{chunk}_offset{offset}_pile_l_tau200_tukey1.npy"
    )
    np.save(destination, Iqo)
    # np.save(f"fIqo_si110_avg_{chunk}_{offset}.npy", fIqo)
    logging.info(f"Finished processing all chunks and beads.")


if __name__ == "__main__":
    main()
