import numpy as np
from sys import argv
import MDAnalysis as mda

file_name = argv[1]
coords = np.load(f'{file_name}.npy')

n_res = coords.shape[0]
n_atoms = n_res

u = mda.Universe.empty(n_atoms, n_residues=n_res, trajectory=True)
u.atoms.positions = coords

with mda.Writer(f'{file_name}.gro') as W:
    W.write(u)