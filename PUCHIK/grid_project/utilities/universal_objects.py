import os
import warnings

import numpy as np
import MDAnalysis as mda
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from PUCHIK.grid_project.utilities.MoleculeSystem import MoleculeSystem

CLUSTER_SEARCH_CUTOFF = 4.7


class ClusterSearch(MoleculeSystem):
    """ A class to search for clusters. The algorithm is somewhat similar to a depth-first search algorithm.
    Select atoms to cluster using select_atoms method and then run find_clusters to return the list of ids in each
    cluster for each frame. !TODO Currently we consider isolated atoms as separate clusters and might need to change it
    """
    def __init__(self, top_path: str, trj_path=None):
        self.ag = None
        if not trj_path:
            self.u = mda.Universe(top_path)
        else:
            self.u = mda.Universe(top_path, trj_path)

    def select_atoms(self, selection: str) -> None:
        self.ag = self.u.select_atoms(selection)

    @staticmethod
    def _cluster_frame(positions: np.ndarray, atom_residue: np.ndarray,
                       n_res: int, resids: np.ndarray) -> list:
        """
        Group residues into connected clusters for a single frame.

        Two residues belong to the same cluster if any pair of their atoms lies
        within ``CLUSTER_SEARCH_CUTOFF`` of each other (no periodic boundary
        conditions, matching the original behaviour). Atom neighbours are found
        with a KD-tree and residues are grouped via connected components, which
        replaces the previous O(n_res^2) recursive depth-first search.

        Args:
            positions (np.ndarray): (n_atoms, 3) atom positions in AtomGroup order.
            atom_residue (np.ndarray): Residue index (0..n_res-1) of each atom.
            n_res (int): Number of residues.
            resids (np.ndarray): Sorted unique resids, indexed by residue index.

        Returns:
            list: Clusters (arrays of resids), ordered by their smallest residue.
        """
        tree = cKDTree(positions)
        pairs = tree.query_pairs(CLUSTER_SEARCH_CUTOFF, output_type='ndarray')

        if len(pairs):
            res_i = atom_residue[pairs[:, 0]]
            res_j = atom_residue[pairs[:, 1]]
        else:
            res_i = res_j = np.empty(0, dtype=int)

        adjacency = coo_matrix(
            (np.ones(len(res_i)), (res_i, res_j)), shape=(n_res, n_res)
        )
        n_components, labels = connected_components(adjacency, directed=False)

        members = [[] for _ in range(n_components)]
        for residue_index, label in enumerate(labels):
            members[label].append(residue_index)
        members.sort(key=lambda group: group[0])

        return [resids[np.array(group)] for group in members]

    def find_clusters(self):
        if self.ag is None:
            raise ValueError('Call select_atoms(...) before find_clusters().')

        n_res = self.ag.n_residues
        resids = np.unique(self.ag.resids)
        n_atoms = self.ag.n_atoms // n_res
        # Residue index of each atom, in AtomGroup order. Assumes a uniform
        # number of atoms per residue, as the original implementation did.
        atom_residue = np.repeat(np.arange(n_res), n_atoms)

        clusters = []
        for _ in self.u.trajectory:
            clusters.append(self._cluster_frame(self.ag.positions, atom_residue, n_res, resids))
        return clusters


def center_to_file(u, selection, o_filename, centroid_pos_selection=None, start=0, skip=1, end=None):
    """
    A utility function to center the system around the center of mass of a selection or an atom.

    :param u: MDAnalysis Universe object
    :param selection: Selection of the section to be centered (protein, nucleic, resname <NANOPARTICLE>, etc)
    :param o_filename: Output trajectory file name
    :param centroid_pos_selection: Selection of the centroid (single atom) positions
    :param start: Starting frame
    :param skip: Number of frames to skip
    :param end: Ending frame
    :return:
    """
    if not o_filename:
        raise TypeError("Please provide an output file name with an extension")

    filename, ext = os.path.splitext(o_filename)

    all_atoms = u.select_atoms('all')

    ag = u.select_atoms(selection)
    try:
        # Check if bond info can be acquired
        _ = ag.fragments
        has_fragments = True
    except Exception:
        has_fragments = False

    if has_fragments:
        transform = mda.transformations.unwrap(ag)
        u.trajectory.add_transformations(transform)
    else:
        warnings.warn("For best result, use a bond-aware format (e.g. tpr) or provide explicit centroid atom selection")

    if centroid_pos_selection:
        centroid = u.select_atoms(centroid_pos_selection)

        if len(centroid) > 1:
            raise ValueError("Centroid selection should be a single atom")
    else:
        centroid = u.select_atoms(selection)

    with mda.Writer(f'{filename}{ext}', all_atoms.n_atoms) as w:

        for _ in u.trajectory[start:end:skip]:
            pbc_dim = u.dimensions[:3]
            all_atom_pos = all_atoms.positions

            if centroid_pos_selection:
                centroid_pos = centroid.positions
            else:
                # Compute center of mass of the selection
                centroid_pos = centroid.center_of_mass()

            new_pos = _translate_system(all_atom_pos, centroid_pos, pbc_dim)

            all_atoms.positions = new_pos

            w.write(all_atoms)


def center_in_memory(u, selection, centroid_pos_selection=None, start=0, skip=1, end=None):
    """
    A utility function to center the system around an atom inplace.

    :param u: MDAnalysis Universe object
    :param selection: Selection of the atom to be centered
    :return:
    """
    u.transfer_to_memory()

    all_atoms = u.select_atoms('all')

    ag = u.select_atoms(selection)
    try:
        # Check if bond info can be acquired
        _ = ag.fragments
        has_fragments = True
    except Exception:
        has_fragments = False

    if has_fragments:
        transform = mda.transformations.unwrap(ag)
        u.trajectory.add_transformations(transform)
    else:
        warnings.warn("For best result, use a bond-aware format (e.g. tpr) or provide explicit centroid atom selection")

    if centroid_pos_selection:
        centroid = u.select_atoms(centroid_pos_selection)

        if len(centroid) > 1:
            raise ValueError("Centroid selection should be a single atom")
    else:
        centroid = u.select_atoms(selection)


    for ts in u.trajectory[start:end:skip]:
        pbc_dim = u.dimensions[:3]
        all_atom_pos = all_atoms.positions

        if centroid_pos_selection:
            centroid_pos = centroid.positions
        else:
            # Compute center of mass of the selection
            centroid_pos = centroid.center_of_mass()

        new_pos = _translate_system(all_atom_pos, centroid_pos, pbc_dim)

        ts.positions = new_pos
    # for ts in u.trajectory:
    #     pbc_dim = u.dimensions[0]
    #     all_atom_pos = all_atoms.positions
    #
    #     center_of_mass_pos = u.select_atoms(selection).center_of_mass()
    #     new_pos = _translate_system(all_atom_pos, center_of_mass_pos, pbc_dim)
    #
    #     ts.positions = new_pos


def _translate_system(positions: np.ndarray, center_atom_pos: np.ndarray, pbc_dim: float) -> np.ndarray:
    """
    A utility function to translate the system towards the center and put all atoms back in the box.

    :param positions: Positions to translate
    :param center_atom_pos: Positions of the new center
    :param pbc_dim: Dimensions of PBC
    :return: New positions
    """
    pbc_center = np.array(pbc_dim) / 2

    # pbc_center = np.array((pbc_dim,) * 3) / 2

    # Translate everything to the desired position
    translation_vec = center_atom_pos - pbc_center
    new_pos = positions - translation_vec

    # Bring atoms back into the PBC
    new_pos = np.mod(new_pos, pbc_dim)

    # u.select_atoms('all').positions = new_pos
    return new_pos
