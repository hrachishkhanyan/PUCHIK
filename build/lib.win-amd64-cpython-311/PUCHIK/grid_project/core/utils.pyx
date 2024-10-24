cimport cython
import numpy as np
cimport numpy as np
from pygel3d import hmesh
from libc.math cimport fabs

# from scipy.spatial import ConvexHull
np.import_array()


def find_distance(hull, np.ndarray points):
    cdef np.ndarray p, res
    cdef float d
    cdef int i, p_length
    # Construct PyGEL Manifold from the convex hull
    m = hmesh.Manifold()
    for s in hull.simplices:
        m.add_face(hull.points[s])

    dist = hmesh.MeshDistance(m)
    p_length = points.shape[0]
    res = np.zeros(p_length)

    for i in range(p_length):
        p = points[i]
        # Get the distance to the point
        # But don't trust its sign, because of possible
        # wrong orientation of mesh face
        d = dist.signed_distance(p)

        # Correct the sign with ray inside test
        if dist.ray_inside_test(p):
            if d > 0:
                d *= -1
        else:
            if d < 0:
                d *= -1

        res[i] = d

    return res


def _is_inside(np.ndarray point, hull, alpha_shape=False) -> bool:
    if alpha_shape:
        vertices = hull.points
        cells = hull.cells
        return point_in_alpha_shape(point, cells, vertices)
    return point_in_hull(point, hull)


def point_in_hull(np.ndarray point, hull):
    cdef double tolerance
    tolerance = 1e-12

    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def point_in_alpha_shape(np.ndarray point, np.ndarray cells, np.ndarray vertices) -> bool:
    cdef np.ndarray cell, tetrahedron_points

    for cell in cells:
        tetrahedron_points = vertices[cell]
        if point_in_tetrahedron(point, tetrahedron_points):
            return True
    return False

def point_in_tetrahedron(np.ndarray point, np.ndarray tetrahedron_points) -> bool:
    cdef np.ndarray v0, v1, v2, v_point
    cdef float denom, u, v, w, t

    v0 = tetrahedron_points[1] - tetrahedron_points[0]
    v1 = tetrahedron_points[2] - tetrahedron_points[0]
    v2 = tetrahedron_points[3] - tetrahedron_points[0]
    v_point = point - tetrahedron_points[0]

    # Calculate determinants
    denom = np.linalg.det(np.column_stack((v0, v1, v2)))
    if denom == 0:
        return False

    # Calculate barycentric coordinates
    u = np.linalg.det(np.column_stack((v_point, v1, v2))) / denom
    v = np.linalg.det(np.column_stack((v0, v_point, v2))) / denom
    w = np.linalg.det(np.column_stack((v0, v1, v_point))) / denom
    t = 1 - u - v - w

    # Check if the point is inside the tetrahedron
    return (u >= 0) and (v >= 0) and (w >= 0) and (t >= 0)