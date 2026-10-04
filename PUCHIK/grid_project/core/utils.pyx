import numpy as np
cimport numpy as np
from pygel3d import hmesh

np.import_array()


def _create_manifold_from_hull(hull):
    m = hmesh.Manifold()

    for s in hull.simplices:
        m.add_face(hull.points[s])
    return m


def find_distance(hull, np.ndarray points):
    cdef np.ndarray d, inside

    dist = hmesh.MeshDistance(_create_manifold_from_hull(hull))

    # Get the distances to all points in one batched call
    # But don't trust their sign, because of possible
    # wrong orientation of mesh faces
    d = np.abs(dist.signed_distance(points))

    # Correct the sign with ray inside test: negative inside, positive outside
    inside = dist.ray_inside_test(points).astype(bool)

    return np.where(inside, -d, d).astype(np.float64)


def points_inside(hull, np.ndarray points, alpha_shape=False):
    """
    Check which points lie inside the hull, in one batched call.

    :param hull: scipy ConvexHull, or AlphaShape if alpha_shape is True
    :param points: (N, 3) array of points, or a single (3,) point
    :param alpha_shape: use a ray inside test against the alpha-shape surface
    :return: (N,) boolean array
    """
    cdef double tolerance = 1e-12
    cdef np.ndarray equations

    points = np.atleast_2d(points)
    if alpha_shape:
        dist = hmesh.MeshDistance(_create_manifold_from_hull(hull))
        return dist.ray_inside_test(points).astype(bool)

    # A point is inside the convex hull if it lies on the inner side of every facet.
    equations = hull.equations
    return np.all(points @ equations[:, :-1].T + equations[:, -1] <= tolerance, axis=1)


def _is_inside(np.ndarray point, hull, alpha_shape=False) -> bool:
    return bool(points_inside(hull, point, alpha_shape)[0])
