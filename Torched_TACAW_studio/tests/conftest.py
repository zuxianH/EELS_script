from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from ase import Atoms
from ase.io.trajectory import Trajectory

@pytest.fixture
def source(tmp_path):
    p = tmp_path/"input.traj"
    with Trajectory(p, "w") as t:
        for i in range(12):
            atoms = Atoms("Si2", positions=[[1+i*.005, 1, 1], [3, 3, 3]],
                          cell=[8, 8, 8], pbc=True, info={"step": i})
            t.write(atoms)
    return p

@pytest.fixture
def settings(source):
    from tacaw_studio.config import defaults
    s = defaults()
    s["input_folder"] = str(source.parent)
    s["inputs"]["files"] = [str(source)]
    s["parameters"].update(trajectory_chunks_size=4, trajectory_chunks_skip_init=0,
                            trajectory_chunks_step=1, trajectory_chunks_nof=1,
                            kspace_shape_full=[16, 16], kspace_ROI_mode="center",
                            kspace_ROI_shape=[8, 8], n_slices=1,
                            frequency_THz_ROI=[-200., 200.])
    return s
