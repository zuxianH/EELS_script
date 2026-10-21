"""Display an XYZ trajectory with Megane in Jupyter.

From a notebook opened in the repository root:
    %run trajectories/visualize_traj.py /path/to/simulation.pos_0.xyz --frame 50
"""

import argparse
from pathlib import Path


def create_viewer(xyz_path, frame_index=50):
    """Return a Megane widget for the supplied trajectory and frame."""
    import megane
    from megane import Pipeline, LoadStructure, LoadTrajectory, AddBonds, Viewport

    xyz_path = str(Path(xyz_path).expanduser().resolve(strict=True))
    pipe = Pipeline()
    structure = pipe.add_node(LoadStructure(xyz_path))
    trajectory = pipe.add_node(LoadTrajectory(xyz=xyz_path))
    bonds = pipe.add_node(AddBonds(source="distance"))
    viewport = pipe.add_node(Viewport(cell_axes_visible=True))

    pipe.add_edge(structure.out.particle, bonds.inp.particle)
    pipe.add_edge(structure.out.particle, trajectory.inp.particle)
    pipe.add_edge(structure.out.particle, viewport.inp.particle)
    pipe.add_edge(structure.out.cell, viewport.inp.cell)
    pipe.add_edge(bonds.out.bond, viewport.inp.bond)
    pipe.add_edge(trajectory.out.traj, viewport.inp.traj)

    viewer = megane.MolecularViewer()
    viewer.set_pipeline(pipe)
    viewer.frame_index = frame_index
    return viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz_path", type=Path, help="input XYZ trajectory")
    parser.add_argument("--frame", type=int, default=50, help="frame index (default: 50)")
    args = parser.parse_args()
    from IPython.display import display

    display(create_viewer(args.xyz_path, args.frame))


if __name__ == "__main__":
    main()
