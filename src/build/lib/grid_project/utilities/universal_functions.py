from copy import deepcopy

from numba import guvectorize, float32, jit, njit
import numpy as np
from scipy.spatial import ConvexHull


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


def make_coordinates(mesh, keep_numbers=False):
    """
    Converts the mesh to coordinates
    Args:
        mesh (np.ndarray):  Mesh to convert into 3D coordinates
        keep_numbers (bool): Resulting tuples will also contain the number of particles at that coordinate if True

    Returns:
        np.ndarray: Ndarray of tuples representing coordinates of each of the points in the mesh
    """
    dim = mesh.ndim
    coords = []

    if dim == 2:
        for i, col in enumerate(mesh):
            for j, elem in enumerate(col):
                if elem > 0:
                    coords.append((i, j)) if not keep_numbers else coords.append((i, j, mesh[i, j]))
    else:
        for i, mat in enumerate(mesh):
            for j, col in enumerate(mat):
                for k, elem in enumerate(col):
                    if elem > 0:
                        coords.append((i, j, k)) if not keep_numbers else coords.append((i, j, k, mesh[i, j, k]))

    return np.array(coords, dtype=int)


def extract_interface(mesh: np.ndarray, rescale_coeff=1):  # TODO change the name: Rescale is not used
    """ Don't need this anymore. See extract_hull """
    original = deepcopy(mesh)

    # Borders of the interface
    min_x = 900
    max_x = 0
    min_y = 900
    max_y = 0
    min_z = 900
    max_z = 0

    for i in range(len(original) - 2):
        for j in range(len(original[i]) - 2):
            for k in range(len(original[i][j]) - 2):
                if original[i][j][k] > 0:
                    if i < min_z:
                        min_z = i
                    if i > max_z:
                        max_z = i
                    if j < min_y:
                        min_y = j
                    if j > max_y:
                        max_y = j
                    if k < min_x:
                        min_x = k
                    if k > max_x:
                        max_x = k
                    if (original[i + 1, j, k] > 0 and original[i - 1, j, k] > 0 and
                            original[i, j + 1, k] > 0 and original[i, j - 1, k] > 0 and
                            original[i, j, k + 1] > 0 and original[i, j, k - 1] > 0):
                        mesh[i, j, k] = 0
    borders = (min_x, max_x, min_y, max_y, min_z, max_z)
    print(borders)
    return mesh, borders


def stretch(a, k, dim=None):
    dim = a.ndim if dim is None else dim
    temp = np.repeat(a, k, axis=0)
    for i in range(1, dim):
        temp = np.repeat(temp, k, i)
    return temp


@guvectorize([(float32[:], float32[:], float32[:])], '(n),(n)->()', target='cuda')
def calculate_distance(p_1, p_2, res):
    x_1, y_1, z_1 = p_1
    x_2, y_2, z_2 = p_2

    res[0] = ((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2 + (z_1 - z_2) ** 2) ** (1 / 2)


@jit(parallel=True)
def distance_array(set_1, set_2):
    res_shape = set_1.shape[0]
    res = np.zeros(shape=(res_shape, res_shape))

    return res


def create_missing_points(matrix, hull):
    result = matrix[hull.vertices].copy()
    for simplex in hull.simplices:
        p_1, p_2 = matrix[simplex]
        x = np.linspace(p_1[0], p_2[0], 10, endpoint=True).round()
        y = np.linspace(p_1[1], p_2[1], 10, endpoint=True).round()

        new_points = np.vstack((x, y)).T
        result = np.vstack((result, new_points))
    return result


def extract_hull(mesh):
    """ Using Qhull algorithm extracts the hull. Utility function create_missing_points adds more points to the hull
    in order to have more continuous surface
    :param mesh: Convex hull for this mesh (ndarray) will be calculated
    :returns result: ndarray of shape mesh.shape, that contains the hull/surface of the original mesh"""

    result = np.zeros(mesh.shape)

    for i, plane in enumerate(mesh):
        coords = make_coordinates(plane)
        if len(coords):
            try:
                hull = ConvexHull(coords)
                coords = create_missing_points(coords, hull).astype(int)
                result[i, coords[:, 0], coords[:, 1]] = 1
                # result[i, coords[hull.vertices][:, 0], coords[hull.vertices][:, 1]] = plane[
                #     coords[hull.vertices][:, 0], coords[hull.vertices][:, 1]]
            except:
                # Don't care
                ...

    return result


def _is_inside(point, mesh):
    x, y, z = point
    yz_proj = mesh.sum(axis=0)
    xz_proj = mesh.sum(axis=1)
    xy_proj = mesh.sum(axis=2)

    if yz_proj[y, z] > 0 and xz_proj[x, z] > 0 and xy_proj[x, y] > 0:
        return True

    return False
