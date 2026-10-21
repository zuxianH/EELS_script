"""Checkpoint ordering, overlap trimming and complete-frame assembly contracts."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from segmented_nvt import assemble, endpoints, latest_checkpoint, trajectory_ranges


class SegmentedTrajectory(unittest.TestCase):
    def test_long_run_endpoints_and_frame_count(self):
        config = dict(equil_steps=10000, prod_steps=405000, segment_steps=50000)
        self.assertEqual(endpoints(config), [50000,100000,150000,200000,250000,300000,350000,400000,415000])
        self.assertEqual(config['prod_steps']//5+1, 81001)

    def test_numeric_checkpoint_order_ignores_empty_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'checkpoints').mkdir()
            for step in (9000,10000):
                (root/'checkpoints'/f'checkpoint.{step}').write_bytes(b'placeholder')
            (root/'checkpoints/checkpoint.11000').touch()
            self.assertEqual(latest_checkpoint(root)[0], 10000)

    def test_interrupted_tail_and_shared_boundaries_are_trimmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'attempts').mkdir()
            config = dict(equil_steps=10, prod_steps=50, dump_every=5)
            for index, (start, stop) in enumerate([(0,30),(30,60),(40,60)]):
                attempt = root/'attempts'/f'{index:04d}'
                attempt.mkdir()
                record = dict(start_step=start, until=stop,
                              trajectory=str((attempt/'production.lammpstrj').relative_to(root)))
                (attempt/'attempt.json').write_text(json.dumps(record))
                with (attempt/'production.lammpstrj').open('w') as dump:
                    for step in range(max(10,start),stop+1,5):
                        dump.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n1\n'
                                   'ITEM: BOX BOUNDS pp pp pp\n0 4\n0 4\n0 4\n'
                                   f'ITEM: ATOMS id type x y z\n1 2 {index} 0 0\n')
            rows = trajectory_ranges(root, config)
            self.assertEqual([(r['start'],r['stop']) for r in rows], [(10,25),(30,35),(40,60)])
            # Old attempt's discarded tail can be truncated at the first
            # rolled-back frame without affecting valid assembly.
            old = root/rows[1]['path']
            data = old.read_text()
            old.write_text(data[:data.index('ITEM: TIMESTEP\n40\n')]+'ITEM: TIMESTEP\n40\n')
            (root/'run_manifest.json').write_text(json.dumps(config|dict(production_frame_count=11)))
            (root/'trajectory_index.json').write_text(json.dumps(dict(completed_step=60, ranges=rows)))
            dest = root/'joined.dump'
            assemble(argparse.Namespace(output=root,destination=dest))
            lines = dest.read_text().splitlines()
            self.assertEqual([int(lines[i+1]) for i,line in enumerate(lines) if line=='ITEM: TIMESTEP'],
                             list(range(10,61,5)))
            self.assertIn('1 2 2 0 0', lines)


if __name__ == '__main__':
    unittest.main()
