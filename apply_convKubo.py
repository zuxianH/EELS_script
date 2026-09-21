#!/usr/bin/env python3
"""Apply the notebook's Kubo detailed-balance correction to raw TACAW arrays.

Run: python3 apply_convKubo.py
Requires NumPy. Defaults match cells 11–13 of plot_paper.ipynb.
Inputs have shape (energy, y, x), with an already FFT-shifted energy axis.
Outputs keep that ordering and multiply each energy plane by
    (E / (kB * T)) / (1 - exp(-E / (kB * T))).
The zero-energy factor is 1. No normalization is applied.
Original files are opened read-only; existing output files are not overwritten.
"""

import argparse
from pathlib import Path

import numpy as np

KB_MEV = 0.08617333262  # meV/K, as in the notebook
PLANCK_J_S = 6.62607015e-34
ELEMENTARY_CHARGE_C = 1.602176634e-19
DEFAULT_FILES = ('Si_10K_beads10_TACAW.npy', 'Si_10K_beads165_TACAW.npy')


def energy_axis(n_frames, timestep_fs, stride):
    """Return the notebook's FFT-shifted energy-loss axis in meV."""
    return (np.fft.fftshift(np.fft.fftfreq(n_frames, d=timestep_fs * stride * 1e-15))
            * PLANCK_J_S / ELEMENTARY_CHARGE_C * 1000)


def kubo_factor(energy_mev, temperature):
    """Stable implementation copied from the notebook's active correction."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive.')
    x = np.asarray(energy_mev, dtype=float) / (KB_MEV * float(temperature))
    factor = np.ones_like(x)
    positive = x > 1e-8
    negative = x < -1e-8
    factor[positive] = x[positive] / (-np.expm1(-x[positive]))
    factor[negative] = x[negative] * np.exp(x[negative]) / np.expm1(x[negative])
    return factor


def convKubo(source, temperature=10.0, timestep_fs=2.5, stride=3):
    """Correct a .npy file in chunks and save it with the suffix _convKubo."""
    source = Path(source)
    for name, value in [('timestep_fs', timestep_fs), ('stride', stride)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    data = np.load(source, mmap_mode='r', allow_pickle=False)
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError(f'{source}: expected nonempty (energy, y, x), got {data.shape}')
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f'{source}: expected numeric data, got {data.dtype}')
    energy = energy_axis(data.shape[0], timestep_fs, stride)
    factors = kubo_factor(energy, temperature)
    destination = source.with_name(source.stem + '_convKubo.npy')
    # Exclusive creation prevents accidentally replacing a previous result.
    with destination.open('xb'):
        pass
    try:
        output = np.lib.format.open_memmap(
            destination, mode='w+', dtype=np.result_type(data.dtype, np.float64),
            shape=data.shape,
        )
        for start in range(0, data.shape[0], 16):
            stop = min(start + 16, data.shape[0])
            block = data[start:stop] * factors[start:stop, None, None]
            if not np.isfinite(block).all():
                raise ValueError(f'{source}: correction produced nonfinite values.')
            output[start:stop] = block
        output.flush()
        del output
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    print(f'Saved {destination}\n'
          f'  shape={data.shape}, T={temperature:g} K, '
          f'energy=[{energy[0]:.6f}, {energy[-1]:.6f}] meV', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', type=Path,
                        help='Input .npy files; defaults to both Si files beside this script.')
    parser.add_argument('--temperature', type=float, default=10.0, help='Kelvin (default: 10).')
    parser.add_argument('--timestep-fs', type=float, default=2.5, help='Default: 2.5 fs.')
    parser.add_argument('--stride', type=int, default=3, help='Default: 3.')
    args = parser.parse_args()
    files = args.files or [Path(__file__).resolve().parent / name for name in DEFAULT_FILES]
    for source in files:
        convKubo(source, args.temperature, args.timestep_fs, args.stride)


if __name__ == '__main__':
    main()
