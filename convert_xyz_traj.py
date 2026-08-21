#import pyms
from ase.io import Trajectory, read, write
import sys
import numpy as np
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt
import logging
import os

""" 
# --- Input arguments ---
beads = int(sys.argv[1])	#total number of beads

# --- Loop over k ---
for k in range(beads):  # adjust range as needed
    logging.info(f"Processing bead index {k}")
    
    input_xyz = f"simulation.pos_{k:01d}.extxyz"
    traj_file = f"simulation.pos_{k:01d}.traj"

    atoms = read(input_xyz, index=":")
    write(traj_file, atoms)

    logging.info(f"Finished processing simulation.pos_{k:01d}.extxyz")

exit()
 """

input_xyz = "simulation.vc.xyz"
traj_file = "simulation.vc.traj"
atoms = read(input_xyz, index=":")
write(traj_file, atoms)
