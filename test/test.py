from matplotlib import pyplot as plt
from densities import Mesh


def test_sphere():
    sphere_pdb = '../files/InP_sphere_r_29.test_structures'
    selection = f'resname UNL or resname SOL and not type H'

    mesh = Mesh(sphere_pdb)
    mesh.select_atoms(selection)

    # print(mesh.dim)
    rescale = 6
    grid_matrix, dens_matrix = mesh.calculate_mesh(rescale=rescale)

    print(mesh.calculate_volume(rescale=rescale))

    coords = mesh.make_coordinates(grid_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

    plt.show()


def test_cylinder():
    sphere_pdb = '../files/InP_cylinder.test_structures'
    selection = f'resname UNL or resname SOL and not type H'

    mesh = Mesh(sphere_pdb)
    mesh.select_atoms(selection)

    # print(mesh.dim)
    rescale = 6
    grid_matrix, dens_matrix = mesh.calculate_mesh(rescale=rescale)

    print(mesh.calculate_volume(rescale=rescale))

    coords = mesh.make_coordinates(grid_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

    plt.show()


if __name__ == '__main__':
    test_cylinder()
    test_sphere()
