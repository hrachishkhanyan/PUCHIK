import numpy as np
cimport numpy as np

np.import_array()


def find_distance(np.ndarray points, np.ndarray mesh_coords, np.ndarray mesh):
    cdef list node_count
    cdef list temp
    cdef dict dists_dict
    cdef double min_dist
    cdef int num
    cdef np.ndarray coord
    cdef bint inside
    cdef double dist

    dists_dict = {}
    node_count = []

    cdef int i
    for i, point in enumerate(points):
        min_dist = 1000  # some starting distance. FIXME take from the existing ones
        coord = point[0:3]
        num = point[3]  # num is the number of particles at node coord
        inside = _is_inside(coord, mesh)  # flag to determine if the point is inside the mesh

        for mesh_point in mesh_coords:
            dist = norm(mesh_point, coord)

            if dist < min_dist:
                min_dist = dist
        if inside:
            min_dist *= -1  # distance from interface is negative if inside

        temp = [min_dist]
        node_count += temp

        if not dists_dict.get(min_dist):  # Check if the distance is in the dictionary
            dists_dict[min_dist] = []  # create it if not

        dists_dict[min_dist] += (temp * num)
        # dists += (temp * num)

    return node_count, dists_dict


def _is_inside(np.ndarray point, np.ndarray mesh):
    cdef int x, y, z
    cdef np.ndarray yz_proj
    cdef np.ndarray xz_proj
    cdef np.ndarray xy_proj
    x, y, z = point
    yz_proj = mesh.sum(axis=0)
    xz_proj = mesh.sum(axis=1)
    xy_proj = mesh.sum(axis=2)

    if yz_proj[y, z] > 0 and xz_proj[x, z] > 0 and xy_proj[x, y] > 0:
        return True

    return False


def norm(np.ndarray p_1, np.ndarray p_2):
    cdef int x_1, y_1, z_1
    cdef int x_2, y_2, z_2
    x_1, y_1, z_1 = p_1
    x_2, y_2, z_2 = p_2

    return ((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2) + (z_1 - z_2) * (z_1 - z_2)) ** (1 / 2)