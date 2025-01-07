cimport cython
import numpy as np
cimport numpy as np
from pygel3d import hmesh
from libc.math cimport fabs

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


def norm(np.ndarray point, np.ndarray plane) -> float:
    cdef np.ndarray p0
    cdef np.ndarray p1
    cdef np.ndarray p2
    cdef np.ndarray normal
    cdef np.ndarray n
    p0, p1, p2 = plane
    normal = np.cross(p1 - p0, p2 - p0)
    n = normal / np.abs(normal)
    dist = np.abs(np.dot(point - p0, n))
    return dist


def _is_inside(np.ndarray point, hull) -> bool:
    return point_in_hull(point, hull)


def point_in_hull(np.ndarray point, hull):
    cdef double tolerance
    tolerance = 1e-12

    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)
