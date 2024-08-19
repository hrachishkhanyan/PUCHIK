import numpy as np
from PUCHIK.grid_project.core.utils import _is_inside


def generate_point(dim):
    """
    Generates a 3D point within dim boundaries
    Args:
        dim (int): Boundaries of the coordinates

    Returns:
        point (np.ndarray): Generated point
    """
    point = np.random.rand(3) * dim
    return point


def monte_carlo(dim, hull, number):
    """
    Monte Carlo volume estimation algorithm

    Args:
        dim (int): Dimensions of the box
        hull (hull object): Convex hull of the structure
        number (int): Number of points to generate

    Returns:
        ratio (float): Ratio of number of points generated inside the volume and overall number of points
    """
    in_volume = 0
    out_volume = 0

    for _ in range(number):
        point = generate_point(dim)

        if _is_inside(point, hull):
            in_volume += 1
        else:
            out_volume += 1

    ratio = in_volume / (out_volume + in_volume)
    return ratio


def monte_carlo_volume(dim, hull, number):
    """
    Utility function responsible for rescaling and calling the actual algorithm

    Args:
        dim (int): Dimensions of the box
        hull (hull object): Hull object
        number (int): Number of points to generate
        rescale (int): Rescale factor

    Returns:
        float: Estimated volume of the structure
    """

    pbc_vol = dim ** 3
    pbc_sys_ratio = monte_carlo(dim, hull, number=number)

    return pbc_sys_ratio * pbc_vol  # , points
