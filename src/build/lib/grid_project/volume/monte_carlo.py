import numpy as np
from ..utilities.universal_functions import check_cube, is_inside


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


def monte_carlo(dim, mesh, number):
    """
    Monte Carlo volume estimation algorithm

    Args:
        dim (int): Dimensions of the box
        number (int): Number of points to generate

    Returns:
        ratio (float): Ratio of number of points generated inside the volume and overall number of points
    """
    in_volume = 0
    out_volume = 0
    # in_points = []
    # out_points = []

    for _ in range(number):
        point = generate_point(dim)
        which_cell = check_cube(*point)

        if is_inside(which_cell, mesh):
            in_volume += 1
            # in_points.append(which_cell)
        else:
            out_volume += 1
            # out_points.append(which_cell)

    ratio = in_volume / (out_volume + in_volume)
    return ratio  #, (in_points, out_points)


def monte_carlo_volume(dim, mesh, number, rescale=None):
    """
    Utility function responsible for rescaling and calling the actual algorithm

    Args:
        number (int): Number of points to generate
        rescale (int): Rescale factor

    Returns:
        float: Estimated volume of the structure
    """

    pbc_vol = dim ** 3
    pbc_sys_ratio = monte_carlo(dim // rescale, mesh, number=number)
    print(pbc_sys_ratio)
    return pbc_sys_ratio * pbc_vol  # , points
