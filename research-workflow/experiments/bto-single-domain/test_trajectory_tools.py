"""Physical regression checks for periodic TiO6 centers and spectra export."""
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np

from trajectory_tools import Frame, convert, read_dump, ti_offsets, vector_summary


def crystal(repeats=(1, 1, 1), offset=0.13, alternating=False, translation=(0, 0, 0)):
    a = np.array([4.0, 4.0, 4.0])
    basis = np.array([[.25,.25,.25], [.75,.75,.75], [.75,.75,.25], [.75,.25,.75], [.25,.75,.75]])
    positions, types = [], []
    for cell in product(*[range(n) for n in repeats]):
        block = (basis + np.array(cell))*a
        block[1, 0] += (-offset if alternating and cell[0] % 2 else offset)
        positions.extend(block)
        types.extend([1,2,3,3,3])
    lengths = a*np.array(repeats)
    positions = (np.array(positions)+np.array(translation)) % lengths
    return Frame(0,np.arange(1,len(types)+1),np.array(types),positions,np.zeros(3),lengths)


class PeriodicDisplacements(unittest.TestCase):
    def test_primitive_cell_counts_distinct_periodic_oxygen_images(self):
        _, offsets, radii = ti_offsets(crystal())
        np.testing.assert_allclose(offsets, [[.13,0,0]], atol=1e-12)
        self.assertLess(radii.max(), 2.2)

    def test_supercell_centers_survive_boundary_crossings(self):
        _, offsets, _ = ti_offsets(crystal((3,3,3), translation=(1.2,2.7,3.8)))
        np.testing.assert_allclose(offsets, np.tile([.13,0,0],(27,1)), atol=1e-12)

    def test_opposite_domains_cancel_mean_but_keep_local_magnitude(self):
        _, offsets, _ = ti_offsets(crystal((4,3,3), alternating=True))
        summary = vector_summary(offsets)
        self.assertAlmostEqual(summary['mean_local_magnitude_A'], .13)
        self.assertAlmostEqual(summary['magnitude_of_mean_A'], 0)
        self.assertAlmostEqual(summary['positive_x_fraction'], .5)

    def test_centrosymmetric_cell_has_zero_displacement(self):
        _, offsets, _ = ti_offsets(crystal(offset=0))
        np.testing.assert_allclose(offsets, 0, atol=1e-12)

    def test_dump_to_ase_preserves_ids_species_cell_timing_and_velocity_units(self):
        from ase.io.trajectory import Trajectory
        from ase.units import fs
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'input.dump'
            output = Path(folder)/'output.traj'
            frame = crystal()
            with source.open('w') as handle:
                for step in (0,10):
                    handle.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n5\n'
                                 'ITEM: BOX BOUNDS pp pp pp\n0 4\n0 4\n0 4\n'
                                 'ITEM: ATOMS id type element x y z vx vy vz ix iy iz\n')
                    for i in reversed(range(5)):
                        x,y,z=frame.positions[i]
                        handle.write(f'{frame.ids[i]} {frame.types[i]} X {x} {y} {z} 1 2 3 0 0 0\n')
            parsed=list(read_dump(source))
            np.testing.assert_array_equal(parsed[0].ids, frame.ids)
            info=convert(source, output, .001)
            self.assertEqual(info['frames'], 2)
            self.assertEqual(info['frame_interval_ps'], .01)
            with Trajectory(str(output)) as trajectory:
                self.assertEqual(trajectory[0].get_chemical_symbols(), ['Ba','Ti','O','O','O'])
                np.testing.assert_allclose(trajectory[0].cell.lengths(), [4,4,4])
                np.testing.assert_allclose(trajectory[0].positions, frame.positions)
                np.testing.assert_allclose(trajectory[0].get_velocities()[0]*1000*fs, [1,2,3])
                self.assertEqual(trajectory[1].info['time_ps'], .01)
                self.assertEqual(trajectory[0].info['lammps_ids'], [1,2,3,4,5])

    def test_export_translates_nonzero_box_origin_to_zero(self):
        from ase.io import read
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.dump'
            output=Path(folder)/'output.xyz'
            source.write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n'
                              'ITEM: BOX BOUNDS pp pp pp\n2 6\n-1 3\n3 7\n'
                              'ITEM: ATOMS id type x y z\n1 2 3 1 6\n')
            convert(source, output, .001)
            atoms=read(output)
            np.testing.assert_allclose(atoms.positions, [[1,2,3]])
            np.testing.assert_allclose(atoms.cell.lengths(), [4,4,4])


if __name__ == '__main__':
    unittest.main()
