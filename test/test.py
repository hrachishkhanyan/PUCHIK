from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt
from mesh.densities import Mesh


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


def timer(f):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        res = f(*args, **kwargs)
        end = perf_counter()
        print(f'Execution time of func "{f.__name__}": {end - start} s')
        return res

    return wrapper


@timer
def add(x, y):
    return x + y

# add_v2 = timer(add)
add(2, 5)

if __name__ == '__main__':
    # test_cylinder()
    # test_sphere()
    pass