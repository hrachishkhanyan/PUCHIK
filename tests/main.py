import numpy as np
from grid_project.utilities.universal_functions import make_coordinates
from matplotlib import pyplot as plt

from grid_project import Mesh

from mol_parts import TYL3_HYDROPHOBIC, TX100_HYDROPHOBIC

DATA_DIR = r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\data\\'


def main(*args, **kwargs):
    rescale = 2
    skip = 1000
    system = ['TY39', 'TX0']
    surf_ratio = '50_50'
    ratio = 0.4
    mesh = Mesh(
        traj=r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\50tyl_50TX\production.part0005_skip_10.xtc',
        top=r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\50tyl_50TX\production.part0005.gro',
        rescale=rescale)
    # mesh = Mesh(r'C:\Users\hrach\Documents\Simulations\TX100\triton_micelle_production999.gro', rescale=rescale)
    # tail = 'C6A C6B C6C C5 C4 C2 C3A C3B C1A C1B C1F C1C C1E C1D'
    # head = 'O1 CA1 CA2 O2 CB1 CB2 O3 CC1 CC2 O4 CD1 CD2 O5 CE1 CE2 O6 CF1 CF2 O7 CG1 CG2 O8 CJ1 CJ2 O11'
    tail = 'C'
    mesh.select_atoms('not type H')
    mesh.select_structure(*system)
    # grid_matrix = mesh.calculate_mesh(rescale=rescale)
    # interface = mesh.calculate_interface()
    # min_z, max_z, min_y, max_y, min_x, max_x = mesh.interface_borders
    # s = mesh.u.select_atoms(
    #     f'not type H and (prop x > {min_x} and prop x < {max_x}) and (prop y > {min_y} and prop y < {max_y})'
    #     f' and (prop z > {min_z} and prop z < {max_z})')
    # s_1 = mesh.u.select_atoms('all')
    # coords = mesh.make_coordinates(interface)
    # outer_mesh = mesh.calculate_mesh('resname SOL and not type H', rescale=rescale)[:, :, :, 0]
    # outer_mesh[min_x:max_x, min_y:max_y, min_z:max_z] = 0  # TODO: This too
    # outer_coords = mesh.make_coordinates(outer_mesh)
    # print(outer_coords)
    # slice_coords = s.positions
    # whole = s_1.positions
    # res = mesh._calc_inner_dens('resname SOL and not type H', coords, interface)
    # np.save(r'C:\Users\hrach\Documents\innter_density_test.npy', [i[0] for i in res[1].values()])
    d, dens = mesh.calculate_density_mp('resname TIP3 and not type H', ratio=ratio, skip=skip)
    d_1, dens_1 = mesh.calculate_density_mp(
        f'resname TY39 and name {TYL3_HYDROPHOBIC} and not type H O', ratio=ratio,
        skip=skip)  # hydrophobic
    d_2, dens_2 = mesh.calculate_density_mp(
        f'resname TY39 and not name {TYL3_HYDROPHOBIC} and not type H O', ratio=ratio,
        skip=skip)  # hydrophilic
    d_3, dens_3 = mesh.calculate_density_mp(
        f'resname TX0 and not name {TX100_HYDROPHOBIC} and not type H O', ratio=ratio,
        skip=skip)  # hydrophobic
    d_4, dens_4 = mesh.calculate_density_mp(
        f'resname TX0 and name {TX100_HYDROPHOBIC} and not type H O', ratio=ratio,
        skip=skip)  # hydrophilic
    np.save(f'{DATA_DIR}/{"_".join(system)}_{surf_ratio}_data_{rescale}_rescaled_{str(ratio).replace("․","_")}.npy',
            np.array([d, dens, d_1, dens_1, d_2, dens_2, d_3, dens_3, d_4, dens_4], dtype=object))
    # d, dens, d_1, dens_1, d_2, dens_2, d_3, dens_3, d_4, dens_4 = np.load(
    #     f'{DATA_DIR}/{"_".join(system)}_{surf_ratio}_data_{rescale}_rescaled.npy', allow_pickle=True)
    # plt.hist([i[0] for i in res[1].values()])

    plt.plot(d, dens)
    plt.plot(d_1, dens_1, label='TY39 Hydrophobic')
    plt.plot(d_2, dens_2, label='TY39 Hydrophilic')
    plt.plot(d_3, dens_3, label='TX100 Hydrophobic')
    plt.plot(d_4, dens_4, label='TX100 Hydrophilic')
    plt.legend()
    plt.grid()
    # plt.ylabel('Number (#/$\AA^3$)')
    # plt.xlabel('Distance from interface (nm)')

    # enclosing_matrix = np.zeros(interface.shape)
    # enclosing_matrix[min_x] = 1
    # enclosing_matrix[max_x] = 1
    # enclosing_matrix[:, min_y] = 1
    # enclosing_matrix[:, max_y] = 1
    # enclosing_matrix[:, :, min_z] = 1
    # enclosing_matrix[:, :, max_z] = 1
    # enclosing_coords = make_coordinates(enclosing_matrix)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d', proj_type='ortho')
    # ax.scatter(coords[::5, 0], coords[::5, 1], coords[::5, 2], color='green', alpha=1)
    # ax.scatter(slice_coords[::20, 0], slice_coords[::20, 1], slice_coords[::20, 2], color='blue', alpha=0.5)
    # ax.scatter(whole[::20, 0], whole[::20, 1], whole[::20, 2], color='yellow', alpha=0.5)
    # ax.scatter(outer_coords[::20, 0], outer_coords[::20, 1], outer_coords[::20, 2], color='red', alpha=0.3)

    # ax.scatter(enclosing_coords[::, 0], enclosing_coords[::, 1], enclosing_coords[::, 2], color='red', alpha=0.1)

    plt.show()


if __name__ == '__main__':
    main()
