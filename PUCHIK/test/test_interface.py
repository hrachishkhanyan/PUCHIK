import pytest
import os
from numpy import isclose
from ..grid_project.core.interface import Interface

TEST_DIR = './PUCHIK/test/test_structures'


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

