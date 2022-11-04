import numpy as np
from grid_project.utilities.universal_functions import make_coordinates
from matplotlib import pyplot as plt

from grid_project import Mesh

from src.grid_project.utilities.universal_functions import stretch


def main(*args, **kwargs):
    rescale = 1
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
    min_z, max_z, min_y, max_y, min_x, max_x = mesh.interface_borders
    s = mesh.u.select_atoms(
        f'not type H and (prop x > {min_x} and prop x < {max_x}) and (prop y > {min_y} and prop y < {max_y})'
        f' and (prop z > {min_z} and prop z < {max_z})')  # TODO: Incorporate this in code
    s_1 = mesh.u.select_atoms('all')
    print(min_z, max_z, min_y, max_y, min_x, max_x)
    coords = mesh.make_coordinates(interface)
    outer_mesh = mesh.calculate_mesh('resname TIP3 and not type H', rescale=rescale)[:, :, :, 0]
    outer_mesh[min_x:max_x, min_y:max_y, min_z:max_z] = 0  # TODO: This too
    outer_coords = mesh.make_coordinates(outer_mesh)
    print(s.positions)
    slice_coords = s.positions
    whole = s_1.positions
    mesh._calc_inner_dens('resname TIP3 and not type H', coords, interface)
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

    enclosing_matrix = np.zeros(interface.shape)
    # enclosing_matrix[min_x] = 1
    # enclosing_matrix[max_x] = 1
    # enclosing_matrix[:, min_y] = 1
    # enclosing_matrix[:, max_y] = 1
    # enclosing_matrix[:, :, min_z] = 1
    # enclosing_matrix[:, :, max_z] = 1
    enclosing_coords = make_coordinates(enclosing_matrix)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')
    # ax.scatter(coords[::5, 0], coords[::5, 1], coords[::5, 2], color='green', alpha=1)
    ax.scatter(slice_coords[::20, 0], slice_coords[::20, 1], slice_coords[::20, 2], color='blue', alpha=0.5)
    # ax.scatter(whole[::20, 0], whole[::20, 1], whole[::20, 2], color='yellow', alpha=0.5)
    ax.scatter(outer_coords[::20, 0], outer_coords[::20, 1], outer_coords[::20, 2], color='red', alpha=0.3)

    # ax.scatter(enclosing_coords[::, 0], enclosing_coords[::, 1], enclosing_coords[::, 2], color='red', alpha=0.1)

    plt.show()


if __name__ == '__main__':
    main()
