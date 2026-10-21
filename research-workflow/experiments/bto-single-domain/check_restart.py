#!/usr/bin/env python3
"""Bounded 40-atom test: compare a continued NVT trajectory with an uninterrupted one."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from trajectory_tools import read_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--cell',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    tool = root/'segmented_nvt.py'
    args.output.mkdir(parents=True,exist_ok=True)
    for name, size, stops in [('split',30,[30,60]), ('whole',60,[60])]:
        output = args.output/name
        subprocess.run([sys.executable,str(tool),'prepare','--output',str(output),
                        '--model',str(args.model),'--cell',str(args.cell),
                        '--replicate','2','2','2','--equil-steps','10','--prod-steps','50',
                        '--segment-steps',str(size),'--checkpoint-every','5'],check=True)
        for stop in stops:
            subprocess.run([sys.executable,str(output/'source/segmented_nvt.py'),'run',
                            '--output',str(output),'--until',str(stop)],check=True)
        subprocess.run([sys.executable,str(output/'source/segmented_nvt.py'),'assemble',
                        '--output',str(output),'--destination',str(output/'joined.dump')],check=True)
    split = list(read_dump(args.output/'split/joined.dump'))
    whole = list(read_dump(args.output/'whole/joined.dump'))
    assert [f.step for f in split] == [f.step for f in whole] == list(range(10,61,5))
    max_position = max_velocity = 0.0
    for a,b in zip(split,whole):
        np.testing.assert_array_equal(a.ids,b.ids)
        np.testing.assert_allclose(a.lengths,b.lengths,rtol=0,atol=1e-12)
        delta = a.positions-b.positions
        delta -= np.rint(delta/a.lengths)*a.lengths
        max_position = max(max_position,float(np.abs(delta).max()))
        max_velocity = max(max_velocity,float(np.abs(a.velocities-b.velocities).max()))
    assert max_position < 1e-9, max_position
    assert max_velocity < 1e-8, max_velocity
    report = dict(status='passed',atoms=40,frames=11,steps=60,
                  max_periodic_position_difference_A=max_position,
                  max_velocity_difference_A_per_ps=max_velocity,
                  criterion='Split vs uninterrupted positions <1e-9 A and velocities <1e-8 A/ps at every saved step',
                  limitation='Short restart continuity test; no thermal or domain convergence evidence')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
