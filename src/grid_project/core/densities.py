# import pickle
# from sys import argv  # for benchmarking only
import warnings
from functools import partial, reduce
from multiprocessing import Pool, cpu_count
# import operator
# from collections import Counter
from MDAnalysis.analysis.distances import self_distance_array
from MDAnalysis.transformations.wrap import wrap
import MDAnalysis as mda
import numpy as np
# from pygel3d import hmesh
# from numpy.linalg import norm

# Local imports
from grid_project.utilities.decorators import timer, logger
from grid_project.volume.monte_carlo import monte_carlo_volume
from grid_project.settings import DEBUG
from grid_project.utilities.universal_functions import extract_hull  # , _is_inside

# if argv[1] == 'cy':
from grid_project.core.utils import find_distance_2  # , norm, _is_inside
# elif argv[1] == 'py':
#    from grid_project.core.pyutils import find_distance_2  # , norm, _is_inside
# from grid_project.core.utils import norm, _is_inside
from scipy.spatial import ConvexHull

np.seterr(invalid='ignore', divide='ignore')

"""
    Grid method for analyzing complex shaped structures
"""
CPU_COUNT = cpu_count()
UNITS = ('nm', 'a')


class Mesh:
    """
        Class creates to create a mesh of points representing different molecule
        types of the system in a grid

        Attributes:
            traj (str): Path to any trajectory format supported by MDAnalysis package
            top (str): Path to any topology format supported by MDAnalysis package. Defaults to None
            rescale (int): Rescales the system down n times. Defaults to 1
    """

    def __init__(self, traj, top=None, rescale=1):
        self.grid_matrix = None
        self.u: mda.Universe = mda.Universe(top, traj) if top else mda.Universe(traj)
        self.ag = None
        self.dim = None
        self.mesh = None
        self.rescale = rescale
        self.interface_rescale = 1  # this is for calculating a rescaled interface then upscaling it
        self.length = self.u.trajectory.n_frames
        self.unique_resnames = None
        self.main_structure = []
        self.interface_borders = None  # defined in calculate_interface method
        self.current_frame = 0

    def select_atoms(self, sel):
        """
        Method for selecting the atoms using MDAnalysis selections

        Args:
            sel (str): selection string

        """
        self.ag = self.u.select_atoms(sel)
        self.unique_resnames = np.unique(self.ag.resnames)
        print('Wrapping trajectory...')
        transform = wrap(self.ag)
        self.u.trajectory.add_transformations(transform)
        print(f'Selected atom group: {self.ag}')

    def select_structure(self, *res_names,
                         auto=False):  # TODO I think this can be determined automatically by clustering
        """
        Use this method to select the structure for density calculations. Enter 1 or more resnames
        :param res_names: Resname(s) of the main structure
        :param auto: Determine automatically if True
        :return: None
        """
        self.main_structure = np.where(np.in1d(self.unique_resnames, res_names))[0]

    def _get_int_dim(self):
        """
        Utility function to get box dimensions

        Returns:
            Dimensions of the box as an int
        """

        return int(np.ceil(self.u.dimensions[0]))

    @logger(DEBUG)
    def calculate_volume(self, number=100_000, units='nm', method='mc', rescale=None):
        """
        Returns the volume of the selected structure

        Args:
            number (int): Number of points to generate for volume estimation
            units (str): Measure unit of the returned value
            method (str): Method of calculation. 'mc' for Monte Carlo estimation,
                'riemann' for Riemann sum method
            rescale (int): Rescale factor

        Returns:
            float: Volume of the structure
        """

        if units not in UNITS:
            raise ValueError('units should be either \'nm\' or \'a\'')

        rescale = rescale if rescale is not None else self.find_min_dist()

        # vol = self._monte_carlo_volume(number, rescale) if method == 'mc' else None
        vol = monte_carlo_volume(self._get_int_dim(), self.grid_matrix, number, rescale) if method == 'mc' else None

        # scale back up and convert from Angstrom to nm if units == 'nm'
        # return vol * self.find_min_dist() ** 3 / 1000 if units == 'nm' else vol * self.find_min_dist() ** 3
        return vol / 1000

        # return vol * rescale ** 3 / 1000 if units == 'nm' else vol * rescale ** 3

    @staticmethod
    def make_grid(pbc_dim: int, dim=1, d4=None) -> np.ndarray:
        """
        Returns a 4D matrix

        Args:
             pbc_dim (int): Dimensions of the box
             dim (int): Dimensions of the box
             d4 (int): Returns an 4-D matrix if d4 is given. 4th dimension contains d4 elements
        """

        x = y = z = pbc_dim // dim + 1
        grid_matrix = np.zeros((x, y, z)) if d4 is None else np.zeros((x, y, z, d4))

        return grid_matrix

    @staticmethod
    def check_cube(x: float, y: float, z: float) -> tuple:
        """
        Find to which cube does the atom belong to
        Args:
            x (float): x coordinate
            y (float): y coordinate
            z (float): z coordinate
            rescale (int): rescale factor

        Returns:
            tuple: Coordinates of the node inside the grid where the point belongs
        """

        # n_x = round(x / rescale_coef)
        # n_y = round(y / rescale_coef)
        # n_z = round(z / rescale_coef)
        n_x = int(x)
        n_y = int(y)
        n_z = int(z)

        return n_x, n_y, n_z

    @staticmethod
    def make_coordinates(mesh, keep_numbers=False):
        """
        Converts the mesh to coordinates
        Args:
            mesh (np.ndarray):  Mesh to convert into 3D coordinates
            keep_numbers (bool): Resulting tuples will also contain the number of particles at that coordinate if True

        Returns:
            np.ndarray: Ndarray of tuples representing coordinates of each of the points in the mesh
        """

        coords = []
        for i, mat in enumerate(mesh):
            for j, col in enumerate(mat):
                for k, elem in enumerate(col):
                    if elem > 0:
                        coords.append((i, j, k)) if not keep_numbers else coords.append((i, j, k, mesh[i, j, k]))

        return np.array(coords, dtype=int)

    def find_min_dist(self):
        """
        Estimate rescale factor.
        Get rid of this.
        Returns:

        """
        return int(np.ceil(self_distance_array(self.ag.positions).min()))

    def _calc_density(self, mol_type, grid_dim, min_distance_coeff):
        """ Not sure what's this for. May delete it later """
        density_matrix = self.make_grid(grid_dim, dim=min_distance_coeff, d4=False)
        for atom in self.ag:
            x, y, z = self.check_cube(*atom.position, rescale=min_distance_coeff)
            if atom.type == mol_type:
                density_matrix[x, y, z] += 1

        return density_matrix

    def _calc_mesh(self, grid_dim, selection, diff=False):
        """
        Calculates the mesh according the atom positions in the box

        Args:
            grid_dim (int): Box dimensions
            rescale: rescale factor
            diff: Is True if we are calculating a mesh for other than the main structure

        Returns:
            np.ndarray: The grid
        """
        self.u.trajectory[self.current_frame]
        atom_group = self.u.select_atoms(selection)

        grid_matrix = self.make_grid(grid_dim, d4=len(self.unique_resnames))
        for atom in atom_group:
            x, y, z = self.check_cube(*atom.position)
            res_number = 0 if not diff else np.where(self.unique_resnames == atom.resname)
            grid_matrix[x, y, z, res_number] += 1

        return grid_matrix

    # @logger(DEBUG)
    def calculate_mesh(self, selection=None, main_structure=False):
        """
        Calculates the mesh using _calc_mesh private method
        Args:
            selection: Selection for atom group to calculate mesh
            rescale: rescale factor
            main_structure (bool): use as the main structure if true (e.g. densities are calculated relative to this)
        Returns:
            np.ndarray: Returns the grid matrix
        """
        # find closest atoms and rescale positions according to this
        # get one dimension

        # print(atom_group.universe.trajectory.frame)
        # self.u.trajectory[atom_group.universe.trajectory.frame]
        # define the matrices

        grid_matrix = self._calc_mesh(self._get_int_dim(), selection, main_structure)

        if main_structure:  # if selection is None, then it's the main structure
            self.grid_matrix = grid_matrix

        return grid_matrix

    # @logger(DEBUG)
    def calculate_interface(self, ratio=0.4, inverse=False):
        """
        Extract the interface from the grid TODO better way needed
        Args:
            inverse (bool): Return everything except for the structure if True
            :param ratio: ratio of moltype/water at a certain point
        Returns:
            np.ndarray: interface matrix

        """

        interface = self.grid_matrix.copy()

        if inverse:
            interface[self.grid_matrix[:, :, :, 1] / self.grid_matrix[:, :, :, 0] >= ratio] = 0
            return interface[:, :, :, 0]

        # The sum(axis=3) is for taking into account the whole structure, which could be constructed of different
        # types of molecules
        interface[(0 < self.grid_matrix[:, :, :, self.main_structure].sum(axis=3) / self.grid_matrix[:, :, :, 0]) & (
                self.grid_matrix[:, :, :, self.main_structure].sum(axis=3) / self.grid_matrix[:, :, :, 0] < ratio)] = 0

        interface = interface[:, :, :, self.main_structure].sum(axis=3)
        # extracted, self.interface_borders = extract_interface(interface, self.interface_rescale)
        interface_hull = extract_hull(interface, 14)
        transposed = extract_hull(interface.T).T  # This is done for filling gaps in the other side
        interface_hull += transposed
        return interface_hull

    # @logger(DEBUG)
    def _normalize_density(self, node_count, dist_dict):
        # Normalization !TODO Generalize
        result = {}
        # dists = np.array(dists)
        unique_dists = set(dist_dict)
        node_count = np.array(node_count)

        for dist in unique_dists:
            dist_length = len(node_count[node_count == dist])
            result[dist] = len(np.array(dist_dict[dist])) / dist_length / (self.rescale ** 3)

        return result

    def _normalize_density_2(self, dists_and_coord, bin_count=12):
        # bin_count - how many equal parts is one dimension divided
        # Տուփը բաժանում ենք N հավասար խորանարդերի։ Հաշվում ենք բոլոր տուփերի միջև դիստանցիաները ու դրանց համապատասխան
        # մասնիկների քանակները։ Մասնիկները քանակը բաժանում ենք խորանարդի ծավալի վրա ու ^ այս դիստանցիաներից յուրաքանչյուրին
        # վերագրում ենք այդ խտությունը
        dimension_extra = 120 - self._get_int_dim()  # TODO: change this
        res_dists = []
        res_densities = []
        number_matrix = np.empty(shape=(self._get_int_dim() + dimension_extra,) * 3 + (2,))
        number_matrix.fill(np.nan)
        bin_size = self._get_int_dim() // bin_count

        for dc in dists_and_coord:
            x, y, z = dc[1]

            number_matrix[x, y, z, 1] = dc[0]

            if np.isnan(number_matrix[x, y, z, 0]):
                number_matrix[x, y, z, 0] = 0
            number_matrix[x, y, z, 0] += 1

        for i in range(0, bin_count):
            for j in range(0, bin_count):
                for k in range(0, bin_count):
                    bin_ = number_matrix[
                           i * bin_size: (i + 1) * bin_size,
                           j * bin_size: (j + 1) * bin_size,
                           k * bin_size: (k + 1) * bin_size
                           ]
                    density = np.nansum(bin_[:, :, :, 0]) / (bin_size ** 3)
                    dists = bin_[:, :, :, 1]
                    temp = list(dists[~np.isnan(dists)])
                    res_dists += temp
                    res_densities += [density] * len(temp)
        res_densities = np.array(res_densities)
        res_dists = np.array(res_dists)

        # Second part
        sort_index = np.argsort(res_dists)
        dens = res_densities[sort_index]
        dist = np.round(res_dists[sort_index])
        min_d, max_d = int(dist.min()), int(dist.max()) + 1  # considering range limit exclusion

        offset = 25  # Array has an offset of 35 to account for negative distances
        dens_fin = np.empty(self._get_int_dim() + dimension_extra + offset)  # Distances are indices of the array.
        dist_fin = np.empty(self._get_int_dim() + dimension_extra + offset)
        dens_fin.fill(np.nan)
        dist_fin.fill(np.nan)

        for j in range(min_d, max_d):
            indices = np.argwhere(dist == j)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'Mean of empty slice')

                dens_fin[offset + j] = dens[indices].mean()

            dist_fin[offset + j] = j
        return dens_fin, dist_fin

    def _extract_from_mesh(self, mol_type):
        if mol_type not in self.unique_resnames:
            raise ValueError(
                f'Molecule type "{mol_type}" is not present in the system. Available types: {self.unique_resnames}'
            )

        mol_index = np.where(self.unique_resnames == mol_type)

        return self.grid_matrix[:, :, :, mol_index]

    def _calc_dens_mp(self, frame_num, selection, interface_selection, norm_bin_count):
        """
        Calculates the density of selection from interface. Multiprocessing version

        Args:
            frame_num (int): Number of the frame
            selection (str): Selection of the atom group density of which is to be calculated
            ratio (float): Ratio moltype/water !TODO for testing. Remove later
        Returns:
            tuple: Density array and corresponding distances
        """
        self.current_frame = frame_num

        mesh_coords = []
        mesh = self.calculate_mesh(selection=interface_selection, main_structure=True)[:, :, :,
               self.main_structure]  # interface = stretch(self.calculate_interface(ratio=ratio), self.interface_rescale, 3)  # uncomment after
        # implementing generalized normalization
        for index, struct in enumerate(self.main_structure):
            mesh_coords.extend(self.make_coordinates(mesh[:, :, :, index]))
        # mesh_coordinates = self.make_coordinates(mesh)
        mesh_coordinates = np.array(mesh_coords)

        selection_mesh = self.calculate_mesh(selection)[:, :, :, 0]

        selection_coords = self.make_coordinates(selection_mesh)

        # selection_coords = self.make_coordinates(selection_mesh, keep_numbers=True)

        # res, d = find_distance_2(selection_coords, mesh_coordinates, interface)  # first method
        try:
            hull = ConvexHull(mesh_coordinates)  # , qhull_options='Q0')
        except Exception as m:
            print('Cannot construct the hull', m)
            return
        res = find_distance_2(hull, selection_coords)  # This and next line are second method
        res, d = self._normalize_density_2(res, bin_count=norm_bin_count)

        return res, d  # Return density and according distance

    # @timer
    def calculate_density(self, selection=None, interface_selection=None, start=0, skip=1, end=None,
                          norm_bin_count=20, cpu_count=CPU_COUNT):
        """
        Calculates density of selection from the interface
        :param end: Final frame
        :param norm_bin_count: Bin count for normalization
        :param cpu_count: Number of cores to use
        :param selection: MDAnalysis selection of ag
        :param interface_selection: Selection of what is considered as interface
        :param start: Starting frame
        :param skip: Skip every n-th frame
        :return:
        """
        n_frames = self.u.trajectory.n_frames if end is None else end

        dens_per_frame = partial(self._calc_dens_mp,
                                 selection=selection,
                                 interface_selection=interface_selection,
                                 norm_bin_count=norm_bin_count)  # _calc_dens_mp function with filled selection using partial
        frame_range = range(start, n_frames, skip)

        with Pool(cpu_count) as worker_pool:
            res = worker_pool.map(dens_per_frame, frame_range)
        res = np.array(res)
        temp_dens = res[:, 0]
        # temp_dist = res[:, 1]  # not needed?

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Mean of empty slice')
            densities = temp_dens.mean(axis=0, where=~np.isnan(temp_dens))  # there will be nan's because of nan's

        densities = np.nan_to_num(densities, nan=0.)  # replacing nan's with 0's

        return densities, np.arange(-25, len(densities) - 25)  # Return densities and according distances

    def interface(self, data=None):
        mesh = self.calculate_interface() if data is None else data
        res = mesh.copy()

        for i, plane in enumerate(res):
            for j, row in enumerate(plane):
                for k, point in enumerate(row):
                    if point > 0:
                        if (mesh[i, j - 1, k] != 0 and mesh[i, j + 1, k] != 0
                                and mesh[i, j, k - 1] != 0 and mesh[i, j, k + 1] != 0
                                and mesh[i - 1, j, k] != 0 and mesh[i + 1, j, k] != 0):
                            res[i, j, k] = 0
        return res


def test(*args, **kwargs):
    from src.test.mol_parts import TYL3_HYDROPHOBIC, TX100_HYDROPHOBIC, TYL7_HYDROPHOBIC, TX100_OXYGEN, TY7_OXYGEN, \
        TY_CARBON, TX100_CARBON
    from grid_project.settings import BASE_DATA_SAVE_DIR
    TRITO_HYDROPHOBIC = 'C19 C20 C21 C22 C23 C24 C25 C26 C27 C28 C29 C30 C31'

    paths = {
        'tx100': r'C:\Users\hrach\Documents\Simulations\TX100',
        'mix': r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3'
    }
    rescale = 1
    skip = 20
    start = 2000
    system = ['TY39']
    interface_selection = \
        f'(resname TY39 and name {TYL3_HYDROPHOBIC}) or (resname TX0 and name {TX100_HYDROPHOBIC}) and not type H O'
    # interface_selection = f'resname TX0 and name {TX100_HYDROPHOBIC} and not type H O'

    # interface_selection = f'resname TRITO and name O1 and not type H'
    name = '100pc_tyl'
    mesh = Mesh(
        traj=fr'{paths["mix"]}\{name}\centered_whole_skip_10.xtc',
        top=fr'{paths["mix"]}\{name}\centered.gro',
        rescale=rescale)


if __name__ == '__main__':
    test()
