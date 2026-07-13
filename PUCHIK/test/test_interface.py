import pytest
import os
import numpy as np
from numpy import isclose
from scipy.spatial import ConvexHull
from PUCHIK.grid_project.core.interface import Interface
from PUCHIK.grid_project.core.AlphaShape import AlphaShape
from PUCHIK.grid_project.core.utils import _is_inside

TEST_DIR = './PUCHIK/test/test_structures'
CYLINDER = os.path.join(TEST_DIR, 'InP_cylinder.pdb')


def test_object_creation():
    m = Interface(
        os.path.join(TEST_DIR, 'InP_cylinder.pdb')
    )


def test_create_mesh():
    m = Interface(
        os.path.join(TEST_DIR, 'InP_cylinder.pdb')
    )

    m.calculate_mesh('resname UNL')


def test_create_hull():
    m = Interface(
        os.path.join(TEST_DIR, 'InP_cylinder.pdb')
    )
    m.select_structure('resname UNL')
    m._create_hull()


def test_calculate_volume():
    m = Interface(
        os.path.join(TEST_DIR, 'InP_cylinder.pdb')
    )
    m.select_structure('resname UNL')
    v = m.calculate_volume()
    assert isclose(v, 146450.0), f'Volume should be close to {146450.0}'


def test_create_alpha_hull():
    m = Interface(
        os.path.join(TEST_DIR, 'InP_cylinder.pdb')
    )
    m.use_alpha_shape = True
    m.select_structure('resname UNL')
    m._create_hull()


# --- Pure discretization helpers -------------------------------------------

def test_make_grid_shape():
    grid = Interface.make_grid(10, dim=1)
    assert grid.shape == (11, 11, 11)
    assert grid.sum() == 0

    grid4d = Interface.make_grid(10, dim=1, d4=3)
    assert grid4d.shape == (11, 11, 11, 3)


def test_check_cube_truncates_toward_zero():
    assert Interface.check_cube(2.9, 0.1, 5.0) == (2, 0, 5)


def test_make_coordinates_recovers_occupied_cells():
    mesh = np.zeros((5, 5, 5))
    mesh[1, 2, 3] = 4
    mesh[0, 0, 0] = 1
    coords = Interface.make_coordinates(mesh)
    assert sorted(map(tuple, coords)) == [(0, 0, 0), (1, 2, 3)]


def test_grid_centers_are_bin_midpoints():
    m = Interface(CYLINDER)
    dim = m._get_int_dim()
    centers = m._grid_centers(bin_count=4)
    assert centers.shape == (4 ** 3, 3)
    step = dim / 4
    # Every coordinate must be a bin midpoint
    expected = {round(step * (i + 0.5), 6) for i in range(4)}
    assert set(np.round(centers[:, 0], 6)) <= expected


# --- Cython point-in-hull test (locks utils.point_in_hull) ------------------

def test_is_inside_convex_hull():
    corners = np.array(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10],
         [10, 10, 0], [10, 0, 10], [0, 10, 10], [10, 10, 10]],
        dtype=float,
    )
    hull = ConvexHull(corners)
    assert _is_inside(np.array([5.0, 5.0, 5.0]), hull, False) is True
    assert _is_inside(np.array([50.0, 50.0, 50.0]), hull, False) is False


# --- Pipeline regression tests (lock discretization/density/count) ----------

def test_calculate_density_regression():
    m = Interface(CYLINDER)
    m.select_atoms('all')
    m.select_structure('resname UNL')
    distances, densities = m.calculate_density('resname UNL', norm_bin_count=10, mp=False)
    # Distances come from the hull discretization, densities from the
    # density-grid binning; both sums lock those code paths.
    assert isclose(np.nansum(distances), -3633.397012233734, rtol=1e-9)
    assert isclose(np.nansum(densities), 27.407407407407405, rtol=1e-9)


def test_mol_count_regression():
    m = Interface(CYLINDER)
    m.select_atoms('all')
    m.select_structure('resname UNL')
    counts = m.mol_count('resname UNL', end=1)
    assert list(np.asarray(counts)) == [5562]


# --- Error handling ---------------------------------------------------------

def test_missing_trajectory_raises():
    with pytest.raises(FileNotFoundError):
        Interface(os.path.join(TEST_DIR, 'does_not_exist.pdb'))


def test_missing_topology_raises():
    with pytest.raises(FileNotFoundError):
        Interface(CYLINDER, top=os.path.join(TEST_DIR, 'does_not_exist.tpr'))


def test_select_atoms_empty_raises():
    m = Interface(CYLINDER)
    with pytest.raises(ValueError):
        m.select_atoms('resname NOPE')


def test_select_structure_empty_raises():
    m = Interface(CYLINDER)
    with pytest.raises(ValueError):
        m.select_structure('resname NOPE')


def test_calculate_density_requires_selection():
    m = Interface(CYLINDER)
    m.select_structure('resname UNL')
    with pytest.raises(ValueError):
        m.calculate_density(None)


def test_mol_count_empty_selection_raises():
    m = Interface(CYLINDER)
    m.select_structure('resname UNL')
    with pytest.raises(ValueError):
        m.mol_count('resname NOPE')


# --- Optional alpha-shape executable ----------------------------------------

def test_is_available_returns_bool():
    assert isinstance(AlphaShape.is_available(), bool)


def test_use_alpha_shape_falls_back_when_executable_missing(monkeypatch):
    monkeypatch.setattr(AlphaShape, 'is_available', classmethod(lambda cls: False))
    m = Interface(CYLINDER)
    with pytest.warns(RuntimeWarning):
        m.use_alpha_shape = True
    assert m.use_alpha_shape is False


def test_use_alpha_shape_enabled_when_executable_present(monkeypatch):
    monkeypatch.setattr(AlphaShape, 'is_available', classmethod(lambda cls: True))
    m = Interface(CYLINDER)
    m.use_alpha_shape = True
    assert m.use_alpha_shape is True


def test_calculate_as_without_executable_raises(monkeypatch):
    monkeypatch.setattr(AlphaShape, '_resolve_executable', staticmethod(lambda: None))
    shape = AlphaShape(np.zeros((4, 3)))
    with pytest.raises(RuntimeError):
        shape.calculate_as(0)
