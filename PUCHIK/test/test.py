from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt

from PUCHIK.grid_project.core.densities import Mesh
from PUCHIK.grid_project.utilities.universal_functions import extract_interface


def test_sphere():
    sphere_pdb = 'test_structures/InP_sphere_r_29.pdb'
    selection = f'resname UNL or resname SOL and not type H'

    mesh = Mesh(sphere_pdb)
    mesh.select_atoms(selection)

    # print(mesh.dim)
    rescale = 3
    grid_matrix = mesh.calculate_mesh(rescale=rescale)

    vol = mesh.calculate_volume(rescale=rescale)
    print(f'Expected volume: 102.160404\nCalculated volume: {vol}')
    # coords = mesh.make_coordinates(grid_matrix)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d', proj_type='ortho')
    #
    # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
    #
    # plt.show()


def test_cylinder():
    sphere_pdb = 'test_structures/InP_cylinder.pdb'
    selection = f'resname UNL or resname SOL and not type H'

    mesh = Mesh(sphere_pdb)
    mesh.select_atoms(selection)

    # print(mesh.dim)
    rescale = 3
    grid_matrix = mesh.calculate_mesh(rescale=rescale)

    vol = mesh.calculate_volume(rescale=rescale)
    print(f'Expected volume: 153.24\nCalculated volume: {vol}')
    # in_points, out_points = points
    # in_points = np.array(in_points)
    # out_points = np.array(out_points)
    # coords = mesh.make_coordinates(grid_matrix)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d', proj_type='ortho')
    #
    # # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
    # ax.scatter(in_points[:, 0], in_points[:, 1], in_points[:, 2], color='green')
    # # ax.scatter(out_points[:, 0], out_points[:, 1], out_points[:, 2], color='red')
    #
    # plt.show()


def test_stretch_cyl():
    sphere_pdb = 'test_structures/InP_cylinder.pdb'
    selection = f'resname UNL or resname SOL and not type H'

    mesh = Mesh(sphere_pdb)
    mesh.select_atoms(selection)

    # print(mesh.dim)
    rescale = 1
    stretch_rescale = 4
    grid_matrix = mesh.calculate_mesh(rescale=rescale)
    grid_matrix_1 = mesh.calculate_mesh(rescale=stretch_rescale)
    interface = extract_interface(grid_matrix)
    interface_1 = extract_interface(grid_matrix_1)

    coords = mesh.make_coordinates(interface)
    coords_1 = mesh.make_coordinates(stretch(interface_1, 4, 3))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')
    step = 10
    ax.scatter(coords[::3, 0], coords[::3, 1], coords[::3, 2])
    ax.scatter(coords_1[::step, 0], coords_1[::step, 1], coords_1[::step, 2], alpha=0.2)

    plt.show()


def test_stretch_micelle():
    mesh = Mesh(r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_7\75tyl_25TX\centered.gro')
    selection = f'resname TY79 TX0 and not type H'

    mesh.select_atoms(selection)
    mesh.select_structure('TY79', 'TX0')

    # print(mesh.dim)
    rescale = 1
    stretch_rescale = 5
    grid_matrix = mesh.calculate_mesh(rescale=rescale)
    grid_matrix_1 = mesh.calculate_mesh(rescale=stretch_rescale)
    interface = extract_interface(grid_matrix[:, :, :, mesh.main_structure].sum(axis=3))
    print(mesh.main_structure)
    interface_1 = extract_interface(grid_matrix_1[:, :, :, mesh.main_structure].sum(axis=3))
    coords = mesh.make_coordinates(interface)
    coords_1 = mesh.make_coordinates(stretch(interface_1, stretch_rescale, 3))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')
    step = 4
    step_1 = 10
    ax.scatter(coords[::step, 0], coords[::step, 1], coords[::step, 2])
    ax.scatter(coords_1[::step_1, 0], coords_1[::step_1, 1], coords_1[::step_1, 2], alpha=0.2)

    plt.show()


def stretch(a, k, dim=None):
    dim = a.ndim if dim is None else dim
    temp = np.repeat(a, k, axis=0)
    for i in range(1, dim):
        temp = np.repeat(temp, k, i)
    return temp


if __name__ == '__main__':
    # test_cylinder()
    # test_sphere()
    test_stretch_micelle()
