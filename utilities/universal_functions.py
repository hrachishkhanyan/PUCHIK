import numpy as np

from utilities.decorators import timer


def check_cube(x: float, y: float, z: float, rescale=1) -> tuple:
    """
    Find to which cube does the atom belong to
    Args:
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
        rescale (int): rescale factor

    Returns:
        tuple: Coordinates of the node inside the grid where the point belongs
    """

    # n_x = round(x / rescale_coef)
    # n_y = round(y / rescale_coef)
    # n_z = round(z / rescale_coef)
    n_x = int(x / rescale)
    n_y = int(y / rescale)
    n_z = int(z / rescale)

    return n_x, n_y, n_z


def is_inside(point, mesh):
    """
    Determines if the point is inside the mesh or not
    :param point: 3D point
    :param mesh: Mesh of points
    :return: bool: True if inside
    """
    dep, row, col = point

    x_mesh_indices = np.where(mesh[:, row, col] > 0)[0]
    y_mesh_indices = np.where(mesh[dep, :, col] > 0)[0]
    z_mesh_indices = np.where(mesh[dep, row, :] > 0)[0]
    # print(point, x_mesh_indices, y_mesh_indices, z_mesh_indices)
    if len(x_mesh_indices) == 0 or len(y_mesh_indices) == 0 or len(z_mesh_indices) == 0:
        return False

    dep_mesh_min, dep_mesh_max = np.where(mesh[:, row, col] > 0)[0].min(), np.where(mesh[:, row, col] > 0)[0].max()
    row_mesh_min, row_mesh_max = np.where(mesh[dep, :, col] > 0)[0].min(), np.where(mesh[dep, :, col] > 0)[0].max()
    col_mesh_min, col_mesh_max = np.where(mesh[dep, row, :] > 0)[0].min(), np.where(mesh[dep, row, :] > 0)[0].max()

    if (row_mesh_min <= row <= row_mesh_max
            and col_mesh_min <= col <= col_mesh_max
            and dep_mesh_min <= dep <= dep_mesh_max):
        return True

    return False