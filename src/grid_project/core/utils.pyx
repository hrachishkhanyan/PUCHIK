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
    cdef double origin_dist
    dists_dict = {}
    node_count = []

    cdef int i
    cdef np.ndarray point

    for i, point in enumerate(points):
        min_dist = 1000  # some starting distance. FIXME take from the existing ones
        coord = point[0:3]
        num = point[3]  # num is the number of particles at node coord
        inside = _is_inside(coord, mesh)  # flag to determine if the point is inside the mesh

        for mesh_point in mesh_coords:
            dist = norm(mesh_point, coord)

            if dist < min_dist:
                min_dist = dist
        origin_dist = norm(coord - (0, 0, 0))

        if inside:
            min_dist *= -1  # distance from interface is negative if inside

        temp = [min_dist]
        node_count += temp

        if not dists_dict.get(min_dist):  # Check if the distance is in the dictionary
            dists_dict[min_dist] = []  # create it if not

        dists_dict[min_dist] += (temp * num)
        # dists += (temp * num)

    return node_count, dists_dict


def find_distance_2(np.ndarray points, np.ndarray mesh_coords, np.ndarray mesh):
    cdef list dists_from_point
    cdef list temp
    cdef double min_dist
    # cdef int num
    cdef np.ndarray coord
    cdef bint inside
    cdef double dist
    cdef double origin_dist
    cdef np.ndarray origin
    dists_and_coord = []  # Will contain distance of the point from the interface and from the origin
    cdef int i
    cdef np.ndarray point
    cdef np.ndarray mesh_point

    for i, point in enumerate(points):
        min_dist = 1000
        coord = point[0:3]
        # num = point[3]  # num is the number of particles at node coord
        inside = _is_inside(coord, mesh)  # flag to determine if the point is inside the mesh

        for mesh_point in mesh_coords:
            dist = norm(mesh_point, coord)

            if dist < min_dist:
                min_dist = dist

        if inside:
            min_dist *= -1

        dists_and_coord.append((min_dist, coord))

    return dists_and_coord

def _is_inside_old(np.ndarray point, np.ndarray mesh):
    """ Doesn't work correctly """
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


def _is_inside(np.ndarray point, np.ndarray mesh):
    cdef int x, y, z
    cdef int x_1, y_1, z_1  # to check if point is inside
    cdef bint x_inside, y_inside, z_inside
    cdef np.ndarray yz_proj
    cdef np.ndarray xz_proj
    cdef np.ndarray xy_proj
    x_inside, y_inside, z_inside = False, False, False
    x, y, z = point
    x_1, y_1, z_1 = point  # initialize to point's coordinates
    yz_proj = mesh.sum(axis=0)
    xz_proj = mesh.sum(axis=1)
    xy_proj = mesh.sum(axis=2)

    # cast a line from the point on yz plane in z direction
    while not y_inside and y_1 < yz_proj.shape[0]:
        if yz_proj[y_1, z] > 0:
            y_inside = True
            break
        y_1 += 1  # go 1 point right in horizontal direction in yz plane

    while not z_inside and z_1 < xz_proj.shape[0]:
        if xz_proj[x, z_1] > 0:
            z_inside = True
            break
        z_1 += 1  # go 1 point right in horizontal direction in xz plane

    while not x_inside and x_1 < xy_proj.shape[0]:
        if xy_proj[x_1, y] > 0:
            x_inside = True
            break
        x_1 += 1  # go 1 point right in horizontal direction in xy plane

    return x_inside and z_inside and y_inside


def norm(np.ndarray p_1, np.ndarray p_2):
    cdef int x_1
    cdef int y_1
    cdef int z_1
    cdef int x_2
    cdef int y_2
    cdef int z_2
    x_1 = p_1[0]
    y_1 = p_1[1]
    z_1 = p_1[2]
    x_2 = p_2[0]
    y_2 = p_2[1]
    z_2 = p_2[2]
    return np.sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2) + (z_1 - z_2) * (z_1 - z_2))




