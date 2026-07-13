import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

from PUCHIK.grid_project.utilities.universal_objects import (
    ClusterSearch,
    CLUSTER_SEARCH_CUTOFF,
)


def _build_pdb(path, positions, atoms_per_res):
    """Write a minimal single-frame PDB with a uniform number of atoms/residue."""
    positions = np.asarray(positions, dtype=float)
    n_atoms = len(positions)
    n_res = n_atoms // atoms_per_res
    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_res,
        atom_resindex=np.repeat(np.arange(n_res), atoms_per_res),
        trajectory=True,
    )
    u.add_TopologyAttr('resid', list(range(1, n_res + 1)))
    u.add_TopologyAttr('resname', ['MOL'] * n_res)
    u.add_TopologyAttr('name', ['A'] * n_atoms)
    u.atoms.positions = positions
    u.atoms.write(str(path))
    return str(path)


def _partition(frame_clusters):
    """Represent a frame's clusters as a set of frozensets of resids (order-agnostic)."""
    return {frozenset(int(r) for r in cluster) for cluster in frame_clusters}


def _brute_force_partition(positions, atoms_per_res, resids):
    """Reference clustering: residues connected if any atom pair is within the
    cutoff (no PBC), grouped with union-find. Mirrors the original definition."""
    n_res = len(positions) // atoms_per_res
    per_res = positions.reshape(n_res, atoms_per_res, 3)

    parent = list(range(n_res))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for a in range(n_res):
        for b in range(a + 1, n_res):
            if (distance_array(per_res[a], per_res[b]) < CLUSTER_SEARCH_CUTOFF).any():
                union(a, b)

    groups = {}
    for i in range(n_res):
        groups.setdefault(find(i), []).append(i)
    return {frozenset(int(resids[i]) for i in members) for members in groups.values()}


def _run(path):
    cs = ClusterSearch(path)
    cs.select_atoms('all')
    return cs.find_clusters()


def test_single_atom_residue_clusters(tmp_path):
    path = _build_pdb(
        tmp_path / 'sys.pdb',
        [[0, 0, 0], [3, 0, 0], [6, 0, 0], [100, 0, 0], [103, 0, 0], [200, 0, 0]],
        atoms_per_res=1,
    )
    result = _run(path)
    assert len(result) == 1  # one frame
    assert _partition(result[0]) == {frozenset({1, 2, 3}), frozenset({4, 5}), frozenset({6})}


def test_connectivity_uses_nearest_atom_not_center(tmp_path):
    # Residue 2's atom at x=3 is close to residue 1, even though their centers
    # are far apart. Clustering must connect them.
    path = _build_pdb(
        tmp_path / 'sys.pdb',
        [[0, 0, 0], [0.5, 0, 0], [10, 0, 0], [3, 0, 0], [50, 0, 0], [50.5, 0, 0]],
        atoms_per_res=2,
    )
    result = _run(path)
    assert _partition(result[0]) == {frozenset({1, 2}), frozenset({3})}


def test_all_residues_isolated(tmp_path):
    path = _build_pdb(
        tmp_path / 'sys.pdb',
        [[0, 0, 0], [100, 0, 0], [200, 0, 0]],
        atoms_per_res=1,
    )
    result = _run(path)
    assert _partition(result[0]) == {frozenset({1}), frozenset({2}), frozenset({3})}


def test_all_residues_connected(tmp_path):
    path = _build_pdb(
        tmp_path / 'sys.pdb',
        [[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]],
        atoms_per_res=1,
    )
    result = _run(path)
    assert _partition(result[0]) == {frozenset({1, 2, 3, 4})}


def test_find_clusters_requires_selection(tmp_path):
    path = _build_pdb(tmp_path / 'sys.pdb', [[0, 0, 0], [1, 0, 0]], atoms_per_res=1)
    cs = ClusterSearch(path)
    with pytest.raises(ValueError):
        cs.find_clusters()


def test_matches_brute_force_reference(tmp_path):
    rng = np.random.default_rng(0)
    for trial in range(15):
        atoms_per_res = int(rng.integers(1, 4))
        n_res = int(rng.integers(2, 12))
        positions = rng.uniform(0, 20, size=(n_res * atoms_per_res, 3))
        path = _build_pdb(tmp_path / f'rand_{trial}.pdb', positions, atoms_per_res)

        result = _run(path)

        # Compare against the reference using positions read back from the PDB
        # (PDB rounds coordinates to 1e-3, so both sides must see the same values).
        u = mda.Universe(path)
        resids = np.unique(u.atoms.resids)
        expected = _brute_force_partition(u.atoms.positions, atoms_per_res, resids)
        assert _partition(result[0]) == expected, f'mismatch on trial {trial}'
