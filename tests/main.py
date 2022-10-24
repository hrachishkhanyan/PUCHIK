import numpy as np
from matplotlib import pyplot as plt

from grid_project import Mesh

from src.grid_project.utilities.universal_functions import stretch


def main(*args, **kwargs):
    rescale = 3
    skip = 2000
    mesh = Mesh(traj=r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_7\50tyl_50TX\centered_whole_skip_20.xtc',
                top=r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_7\50tyl_50TX\centered.gro', rescale=rescale)

    # tail = 'C6A C6B C6C C5 C4 C2 C3A C3B C1A C1B C1F C1C C1E C1D'
    # head = 'O1 CA1 CA2 O2 CB1 CB2 O3 CC1 CC2 O4 CD1 CD2 O5 CE1 CE2 O6 CF1 CF2 O7 CG1 CG2 O8 CJ1 CJ2 O11'
    tail = 'C'
    mesh.select_atoms('not type H')
    mesh.select_structure('TY79', 'TX0')
    grid_matrix = mesh.calculate_mesh(rescale=rescale)
    interface = mesh.calculate_interface()
    coords = mesh.make_coordinates(stretch(interface, 4, 3))
    # d, dens = mesh.calculate_density_mp('resname TIP3 and not type H', skip=skip)
    # d_1, dens_1 = mesh.calculate_density_mp(f'resname TX0 and type C and not type H', skip=skip)
    # d_2, dens_2 = mesh.calculate_density_mp(f'resname TY79 and type C and not type H', skip=skip)
    # np.save(f'../src/test/TY79_TX0_50_50_data_{rescale}.npy', np.array([d, dens, d_1, dens_1, d_2, dens_2]))
    # d, dens, d_1, dens_1, d_2, dens_2 = np.load('../src/test/TY79_TX0_50_50_data.npy', allow_pickle=True)
    # plt.plot(d, dens)
    # plt.plot(d_1, dens_1)
    # plt.plot(d_2, dens_2)
    #
    # plt.ylabel('Number (#/$\AA^3$)')
    # plt.xlabel('Distance from interface (nm)')


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')
    ax.scatter(coords[::10, 0], coords[::10, 1], coords[::10, 2], color='green', alpha=0.5)

    plt.show()


if __name__ == '__main__':
    main()
