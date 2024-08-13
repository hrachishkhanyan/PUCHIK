import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
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

    def _dfs(self, index: int, positions: np.ndarray, cluster: list, checked: list) -> list:
        checked.append(index)

        for i in range(len(positions)):

            if i != index and i not in checked:
                distances = distance_array(positions[index], positions[i])

                if (distances < CLUSTER_SEARCH_CUTOFF).any():
                    cluster.append(i)
                    self._dfs(i, positions, cluster, checked)

        return cluster

    def find_clusters(self):
        n_res = self.ag.n_residues
        resids = np.unique(self.ag.resids)
        n_atoms = self.ag.n_atoms // n_res
        clusters = []

        positions = np.zeros((len(self.u.trajectory), n_res, n_atoms, 3))
        for ts in self.u.trajectory:
            current_frame_clusters = []
            checked_residues = []

            for i in range(n_res):
                for j in range(n_atoms):
                    positions[ts.frame, i, j] = self.ag[i * n_atoms + j].position

            for i, position in enumerate(positions[ts.frame]):
                if i not in checked_residues:
                    current_frame_clusters.append(resids[self._dfs(i, positions[ts.frame], [i], checked_residues)])
            clusters.append(current_frame_clusters)
        return clusters


def center_to_file(u, selection, o_filename=None, ext=None):
    """
    A utility function to center the system around an atom.

    :param u: MDAnalysis Universe object
    :param selection: Selection of the atom to be centered
    :param o_filename: Output trajectory file name
    :param ext: Extension of the output trajectory file
    :return:
    """
    all_atoms = u.select_atoms('all')

    with mda.Writer(f'{o_filename}.xtc', all_atoms.n_atoms) as w:
        for _ in u.trajectory:
            pbc_dim = u.dimensions[0]
            all_atom_pos = all_atoms.positions
            center_atom_pos = u.select_atoms(selection).positions
            new_pos = _translate_system(all_atom_pos, center_atom_pos, pbc_dim)
            all_atoms.positions = new_pos
            w.write(all_atoms)


def center_in_memory(u, selection):
    """
    A utility function to center the system around an atom inplace.

    :param u: MDAnalysis Universe object
    :param selection: Selection of the atom to be centered
    :return:
    """
    u.transfer_to_memory()

    all_atoms = u.select_atoms('all')

    for ts in u.trajectory:
        pbc_dim = u.dimensions[0]
        all_atom_pos = all_atoms.positions
        center_atom_pos = u.select_atoms(selection).positions

        new_pos = _translate_system(all_atom_pos, center_atom_pos, pbc_dim)

        ts.positions = new_pos


def _translate_system(positions: np.ndarray, center_atom_pos: np.ndarray, pbc_dim: float) -> np.ndarray:
    """
    A utility function to translate the system towards the center and put all atoms back in the box.

    :param positions: Positions to translate
    :param center_atom_pos: Positions of the new center
    :param pbc_dim: Dimensions of PBC
    :return: New positions
    """
    pbc_center = np.array((pbc_dim,) * 3) / 2

    # Translate everything to the desired position
    translation_vec = center_atom_pos - pbc_center
    new_pos = positions - translation_vec

    # Bring atoms back into the PBC
    new_pos = np.mod(new_pos, pbc_dim)

    # u.select_atoms('all').positions = new_pos
    return new_pos
