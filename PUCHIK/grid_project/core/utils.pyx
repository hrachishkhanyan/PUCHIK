import numpy as np
cimport numpy as np
from pygel3d import hmesh
# from scipy.spatial import ConvexHull

np.import_array()

def find_distance(hull, np.ndarray points):
    cdef list res
    cdef np.ndarray p
    cdef float d

    # Construct PyGEL Manifold from the convex hull
    m = hmesh.Manifold()
    for s in hull.simplices:
        m.add_face(hull.points[s])

    dist = hmesh.MeshDistance(m)
    res = []
    for p in points:
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

        res.append(d)

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

def make_coordinates(mesh, keep_numbers=False):
    """
    Converts the mesh to coordinates
    Args:
        mesh (np.ndarray):  Mesh to convert into 3D coordinates
        keep_numbers (bool): Resulting tuples will also contain the number of particles at that coordinate if True

    Returns:
        np.ndarray: Ndarray of tuples representing coordinates of each of the points in the mesh
    """
    cdef int dim
    cdef list coords

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

