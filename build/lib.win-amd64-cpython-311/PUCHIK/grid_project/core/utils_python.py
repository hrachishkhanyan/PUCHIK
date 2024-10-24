import numpy as np
from pygel3d import hmesh


def find_distance(hull, points):
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


def _is_inside(point, hull) -> bool:
    return point_in_hull(point, hull)


def point_in_hull(point, hull):
    tolerance = 1e-12

    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)
