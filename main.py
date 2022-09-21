import numpy as np
from matplotlib import pyplot as plt

from src.grid_project.densities import Mesh


def main(*args, **kwargs):
    rescale = 3
    skip = 500
    mesh = Mesh(traj=r'C:\Users\hrach\Documents\Simulations\TX114\production_pure.xtc',
                top=r'C:\Users\hrach\Documents\Simulations\TX114\pure\production.gro', rescale=rescale)
    tail = 'C6A C6B C6C C5 C4 C2 C3A C3B C1A C1B C1F C1C C1E C1D'
    head = 'O1 CA1 CA2 O2 CB1 CB2 O3 CC1 CC2 O4 CD1 CD2 O5 CE1 CE2 O6 CF1 CF2 O7 CG1 CG2 O8 CJ1 CJ2 O11'

    mesh.select_atoms('not type H')
    mesh.select_structure('TX4')

    d, dens = mesh.calculate_density_mp('resname SOL and not type H', skip=skip)
    d_1, dens_1 = mesh.calculate_density_mp(f'resname TX4 and name {tail} and not type H', skip=skip)
    d_2, dens_2 = mesh.calculate_density_mp(f'resname TX4 and name {head} and not type H', skip=skip)
    np.save('TX114_data.npy', np.array([d_1, dens_1, d_2, dens_2]))
    # plt.plot(d, dens)
    plt.plot(d_1, dens_1)
    plt.plot(d_2, dens_2)

    plt.ylabel('Number (#/$\AA^3$)')
    plt.xlabel('Distance from interface (nm)')

    plt.show()


if __name__ == '__main__':
    main()
