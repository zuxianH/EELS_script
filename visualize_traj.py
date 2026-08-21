import megane
from megane import Pipeline, LoadStructure, LoadTrajectory, AddBonds, Viewport

xyz_path = "/home/zuxian/Documents/PhD/STEM_EELS/PlotRegions/simulation.pos_0.xyz"

pipe = Pipeline()

s = pipe.add_node(LoadStructure(xyz_path))
traj = pipe.add_node(LoadTrajectory(xyz=xyz_path))
ab = pipe.add_node(AddBonds(source="distance"))
v = pipe.add_node(Viewport(cell_axes_visible=True))

pipe.add_edge(s.out.particle, ab.inp.particle)
pipe.add_edge(s.out.particle, traj.inp.particle)
pipe.add_edge(s.out.particle, v.inp.particle)
pipe.add_edge(s.out.cell, v.inp.cell)
pipe.add_edge(ab.out.bond, v.inp.bond)
pipe.add_edge(traj.out.traj, v.inp.traj)

viewer = megane.MolecularViewer()
viewer.set_pipeline(pipe)

viewer.frame_index = 50
viewer
