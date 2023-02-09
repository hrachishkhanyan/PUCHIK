import pickle
from sys import argv  # for benchmarking only
import warnings
from functools import partial, reduce
from multiprocessing import Pool, cpu_count
import operator
from collections import Counter
from MDAnalysis.analysis.distances import self_distance_array
from MDAnalysis.transformations.wrap import wrap
import MDAnalysis as mda
import numpy as np
from pygel3d import hmesh
# from numpy.linalg import norm

# Local imports
from grid_project.utilities.decorators import timer, logger
from grid_project.volume.monte_carlo import monte_carlo_volume
from grid_project.settings import DEBUG
from grid_project.utilities.universal_functions import extract_hull  # , _is_inside

# if argv[1] == 'cy':
from grid_project.core.utils import find_distance_2  # , norm, _is_inside
# elif argv[1] == 'py':
#     from grid_project.core.pyutils import find_distance_2  # , norm, _is_inside

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
        self.dim = self.u.dimensions
        self.mesh = None
        self.rescale = rescale
        self.interface_rescale = 1  # this is for calculating a rescaled interface then upscaling it
        self.length = self.u.trajectory.n_frames
        self.unique_resnames = None
        self.main_structure = []
        self.interface_borders = None  # defined in calculate_interface method

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
        return int(np.ceil(self.u.trajectory[0].dimensions[0]))

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
    def check_cube(x: float, y: float, z: float, rescale=1) -> tuple:
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
        n_x = int(x / rescale)
        n_y = int(y / rescale)
        n_z = int(z / rescale)

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
            # print(atom.position)
            x, y, z = self.check_cube(*atom.position, rescale=min_distance_coeff)
            if atom.type == mol_type:
                density_matrix[x, y, z] += 1

        return density_matrix

    def _calc_mesh(self, grid_dim, rescale, ag, diff=False):
        """
        Calculates the mesh according the atom positions in the box

        Args:
            grid_dim (int): Box dimensions
            rescale: rescale factor
            diff: Is True if we are calculating a mesh for other than the main structure

        Returns:
            np.ndarray: The grid
        """
        grid_matrix = self.make_grid(grid_dim, dim=rescale, d4=len(self.unique_resnames))

        if ag is None:
            ag = self.ag

        for atom in ag:
            x, y, z = self.check_cube(*atom.position, rescale=rescale)
            res_number = 0 if not diff else np.where(self.unique_resnames == atom.resname)
            grid_matrix[x, y, z, res_number] += 1

        return grid_matrix

    # @logger(DEBUG)
    def calculate_mesh(self, selection=None, main_structure=False, rescale=None):
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
        grid_dim = self._get_int_dim()  # get one dimension
        atom_group = self.u.select_atoms(selection) if selection is not None else self.ag

        # define the matrices
        grid_matrix = self._calc_mesh(grid_dim, rescale, atom_group, main_structure)

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

    # def find_distance_2(self, points, mesh_coords):
    #
    #     dists_and_coord = []  # Will contain distance of the point from the interface and from the origin
    #     hull = ConvexHull(mesh_coords, qhull_options='Q0')
    #
    #     for i, point in enumerate(points):
    #         min_dist = 1000
    #         coord = point[0:3]
    #         # num = point[3]  # num is the number of particles at node coord
    #         inside = _is_inside(coord, hull)  # flag to determine if the point is inside the mesh
    #
    #         for simplex in hull.simplices:
    #             # Calculate distance between centroid of simplices and coordinates
    #             dist = self.norm(coord, mesh_coords[simplex])
    #
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 simp = simplex
    #         # for vertex in hull.vertices:
    #         #     # Calculate distances between vertices and coordinates
    #         #     dist = norm(coord, mesh_coords[vertex])
    #         #
    #         #     if dist < min_dist:
    #         #         min_dist = dist
    #
    #         sign = -1 if inside else 1
    #         # if min_dist > 6:
    #         #     print(coord)
    #         dists_and_coord.append((sign * min_dist, coord))
    #         # dists_and_coord.append((min_dist, coord))
    #
    #     return dists_and_coord

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

        res_dists = []
        res_densities = []
        number_matrix = np.empty(shape=(self._get_int_dim(),) * 3 + (2,))
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
        dens_fin = np.empty(self._get_int_dim())  # Distances are indices of the array.
        dist_fin = np.empty(self._get_int_dim())
        dens_fin.fill(np.nan)
        dist_fin.fill(np.nan)
        offset = 25  # Array has an offset of 25 to account for negative distances

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

    def calculate_density(self, interface_selection, selection=None, start=0, skip=1, number_of_frames=None,
                          norm_bin_count=20):
        """
        Calculates the density of selection from interface

        Args:
            selection (str): Selection of the atom group density of which is to be calculated
            skip (int): Use every 'skip'-th frame to calculate density
            interface_selection (str):
            number_of_frames: For profiling only
            norm_bin_count: defines how many bins is the mesh divided to during normalization
        Returns:
            tuple: Density array and corresponding distances

        """

        frame_count = self.length // skip if number_of_frames is None else number_of_frames
        res = [None] * frame_count
        dist = [None] * frame_count
        # dists_dict_list = [{}] * (frame_count + 1)
        j = 0  # index of the frame
        loop_range = range(0, frame_count) if number_of_frames is None else np.linspace(0, self.length,
                                                                                        number_of_frames,
                                                                                        dtype=int,
                                                                                        endpoint=False)

        for i in loop_range:
            frame_num = i * skip if number_of_frames is None else i
            self.u.trajectory[frame_num]
            mesh_coords = []

            mesh = self.calculate_mesh(selection=interface_selection, main_structure=True,
                                       rescale=self.interface_rescale)[:, :, :, self.main_structure]
            # interface = self.calculate_interface()

            # mesh_coordinates = self.make_coordinates(interface)
            for index, _ in enumerate(self.main_structure):
                mesh_coords.extend(self.make_coordinates(mesh[:, :, :, index]))
            mesh_coordinates = np.array(mesh_coords)

            selection_mesh = self.calculate_mesh(selection, rescale=self.rescale)[:, :, :, 0]
            selection_coords = self.make_coordinates(selection_mesh)
            try:
                hull = ConvexHull(mesh_coordinates)  # , qhull_options='Q0')
            except:
                print('Cannot construct the hull')
                return
            # res[index] = find_distance_2(selection_coords, mesh_coordinates, interface)
            print(j)
            res[j] = find_distance_2(hull, selection_coords)

            res[j], dist[j] = self._normalize_density_2(res[j], bin_count=norm_bin_count)
            j += 1

        # This can be a general function
        res = np.array(res)
        temp_dens = res[:, 0]
        # temp_dist = res[:, 1]  # not needed?

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Mean of empty slice')
            densities = temp_dens.mean(axis=0, where=~np.isnan(temp_dens))  # there will be nan's because of nan's
        # distances = temp_dist.mean(axis=0, where=~np.isnan(temp_dist))

        densities = np.nan_to_num(densities, nan=0.)

        return densities, np.arange(-25, self._get_int_dim() - 25)  # Return densities and according distances

    def _calc_dens_mp(self, frame_num, selection, interface_selection, ratio, norm_bin_count):
        """
        Calculates the density of selection from interface. Multiprocessing version

        Args:
            frame_num (int): Number of the frame
            selection (str): Selection of the atom group density of which is to be calculated
            ratio (float): Ratio moltype/water !TODO for testing. Remove later
        Returns:
            tuple: Density array and corresponding distances
        """

        self.u.trajectory[frame_num]
        mesh_coords = []
        mesh = self.calculate_mesh(selection=interface_selection, main_structure=True,
                                   rescale=self.interface_rescale)[:, :, :,
               self.main_structure]  # interface = stretch(self.calculate_interface(ratio=ratio), self.interface_rescale, 3)  # uncomment after
        # implementing generalized normalization
        for index, struct in enumerate(self.main_structure):
            mesh_coords.extend(self.make_coordinates(mesh[:, :, :, index]))
        # mesh_coordinates = self.make_coordinates(mesh)
        mesh_coordinates = np.array(mesh_coords)

        # inverse = self.calculate_mesh(selection, rescale=self.rescale)[:, :, :, 0]
        selection_mesh = self.calculate_mesh(selection, rescale=self.rescale)[:, :, :, 0]

        selection_coords = self.make_coordinates(selection_mesh)
        # selection_coords = self.make_coordinates(selection_mesh, keep_numbers=True)

        # res, d = find_distance_2(selection_coords, mesh_coordinates, interface)  # first method
        try:
            hull = ConvexHull(mesh_coordinates)  # , qhull_options='Q0')
        except:
            print('Cannot construct the hull')
            return
        res = find_distance_2(hull, selection_coords)  # This and next line are second method
        res, d = self._normalize_density_2(res, bin_count=norm_bin_count)

        return res, d  # Return density and according distance

    # @timer
    def calculate_density_mp(self, selection=None, interface_selection=None, ratio=0.4, start=0, skip=1, end=None,
                             norm_bin_count=20, cpu_count=CPU_COUNT):
        """
        Calculates density of selection from the interface
        :param selection: MDAnalysis selection of ag
        :param interface_selection: Selection of what is considered as interface
        :param ratio:
        :param start: Starting frame
        :param skip: Skip every n-th frame
        :return:
        """
        n_frames = self.u.trajectory.n_frames if end is None else end

        dens_per_frame = partial(self._calc_dens_mp,
                                 selection=selection,
                                 interface_selection=interface_selection,
                                 ratio=ratio,
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
        # distances = temp_dist.mean(axis=0, where=~np.isnan(temp_dist))

        densities = np.nan_to_num(densities, nan=0.)  # replacing nan's with 0's
        # distances = np.nan_to_num(distances, nan=0.)
        # with open(r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\data\\water_rescale_1_new.pickle',
        #           'wb') as file:
        #     pickle.dump(res, file)
        """  First method
        
        densities = [elem[0] for elem in res]
        dists = [elem[1] for elem in res]
        dists_dict_list = [self._normalize_density(densities[index], d) for index, d in enumerate(dists)]

        res = dict(reduce(operator.add, map(Counter, dists_dict_list)))

        res = {k: res[k] for k in sorted(res)}
        res = np.array(list(res.keys())), np.array(list(res.values())) / len(frame_range)
        return res
        """
        return densities, np.arange(-25, self._get_int_dim() - 25)  # Return densities and according distances

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


def main(*args, **kwargs):
    from src.test.mol_parts import TYL3_HYDROPHOBIC, TX100_HYDROPHOBIC, TYL7_HYDROPHOBIC, TX100_OXYGEN, TY7_OXYGEN, \
        TY_CARBON, TX100_CARBON
    from grid_project.settings import BASE_DATA_SAVE_DIR
    TRITO_HYDROPHOBIC = 'C19 C20 C21 C22 C23 C24 C25 C26 C27 C28 C29 C30 C31'

    paths = {
        'tx100': r'C:\Users\hrach\Documents\Simulations\TX100',
        'mix': r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_7'
    }
    rescale = 1
    skip = 2000
    start = 2000
    system = ['TY79', 'TX0']
    # system = ['TRITO']

    interface_selection = f'(resname TY79 and name {TYL7_HYDROPHOBIC}) or (resname TX0 and name {TX100_HYDROPHOBIC}) and not type H O'
    # interface_selection = f'resname TX0 and name {TX100_HYDROPHOBIC} and not type H O'

    # interface_selection = f'resname TRITO and name O1 and not type H'
    surf_ratio = 'NA'
    ratio = 0.6
    mesh = Mesh(
        traj=fr'{paths["mix"]}\25tyl_75TX\centered_whole_skip_10.xtc',
        top=fr'{paths["mix"]}\25tyl_75TX\centered.gro',
        rescale=rescale)
    # mesh = Mesh(
    #     traj=fr'{paths["tx100"]}\\triton_micelle_production.xtc',
    #     top=fr'{paths["tx100"]}\\triton_micelle_production999.gro',
    #     rescale=rescale)

    mesh.interface_rescale = rescale
    mesh.select_atoms('not type H')
    mesh.select_structure(*system)

    # dens, dist = mesh.calculate_density_mp(selection='resname SOL and not type H',
    #                                        interface_selection=interface_selection,
    #                                        start=start,
    #                                        ratio=ratio,
    #                                        skip=skip
    #                                        )

    # dens_1, dist_1 = mesh.calculate_density_mp(
    #     f'resname TRITO and not name {TRITO_HYDROPHOBIC} and not type H O', interface_selection=interface_selection,
    #     start=start,
    #     ratio=ratio,
    #     skip=skip)  # hydrophilic
    # dens_2, dist_2 = mesh.calculate_density_mp(
    #     f'resname TRITO and name {TRITO_HYDROPHOBIC} and not type H O', interface_selection=interface_selection,
    #     start=start,
    #     ratio=ratio,
    #     skip=skip)  # hydrophobic

    # dens, dist = mesh.calculate_density_mp(selection='resname TIP3 and not type H',
    #                                        interface_selection=interface_selection, skip=skip)  # , number_of_frames=1)
    #
    dens_1, dist_1 = mesh.calculate_density(selection=f'resname TY79 and not name {TYL7_HYDROPHOBIC} and not type H O',
                                            interface_selection=interface_selection,
                                            start=start,
                                            number_of_frames=1,
                                            skip=skip)  # hydrophilic
    # dens_2, dist_2 = mesh.calculate_density_mp(
    #     f'resname TY79 and name {TYL7_HYDROPHOBIC} and not type H O', interface_selection=interface_selection,
    #     start=start,
    #     ratio=ratio,
    #     skip=skip)  # hydrophobic
    # dens_3, dist_3 = mesh.calculate_density_mp(
    #     f'resname TX0 and not name {TX100_HYDROPHOBIC} and not type H O', interface_selection=interface_selection,
    #     start=start,
    #     ratio=ratio,
    #     skip=skip)  # hydrophilic
    # dens_4, dist_4 = mesh.calculate_density_mp(
    #     f'resname TX0 and name {TX100_HYDROPHOBIC} and not type H O', interface_selection=interface_selection,
    #     start=start,
    #     ratio=ratio,
    #     skip=skip)  # hydrophobic

    # np.save(
    #     f'{BASE_DATA_SAVE_DIR}/{"_".join(system)}_data_{rescale}_rescaled_{str(ratio).replace("․", "_")}.npy',
    #     np.array([dens, dist, dens_1, dist_1, dens_2, dist_2, dens_3, dist_3, dens_4, dist_4], dtype=object))  # , d_3, dens_3, d_4, dens_4], dtype=object))
    # d, dens, d_1, dens_1, d_2, dens_2, d_3, dens_3, d_4, dens_4 = np.load(
    #     f'{DATA_DIR}/{"_".join(system)}_{surf_ratio}_data_{rescale}_rescaled.npy', allow_pickle=True)
    # plt.hist([i[0] for i in res[1].values()])

    from matplotlib import pyplot as plt
    # plt.plot(dist, dens, label='Water')

    plt.plot(dist_1[:-1], dens_1, label='TY7 hydrophilic')
    plt.plot(dist_2[:-1], dens_2, label='TY7 hydrophobic')
    plt.plot(dist_3[:-1], dens_3, label='TX0 hydrophilic')
    plt.plot(dist_4[:-1], dens_4, label='TX0 hydrophobic')
    plt.legend()
    plt.xlim(-15, 50)
    # plt.plot(dist_1, dens_1)
    #
    # plt.plot(dist_2, dens_2)
    # plt.plot(d_1, dens_1, label='TRITO Hydrophobic')
    # plt.plot(d_2, dens_2, label='TRITO Hydrophilic')
    # plt.plot(d_3, dens_3, label='TX100 Hydrophobic')
    # plt.plot(d_4, dens_4, label='TX100 Hydrophilic')
    # plt.legend()
    # plt.grid()
    # plt.ylabel('Number (#/$\AA^3$)')
    # plt.xlabel('Distance from interface (nm)')

    # enclosing_matrix = np.zeros(interface.shape)
    # enclosing_matrix[min_x] = 1
    # enclosing_matrix[max_x] = 1
    # enclosing_matrix[:, min_y] = 1
    # enclosing_matrix[:, max_y] = 1
    # enclosing_matrix[:, :, min_z] = 1
    # enclosing_matrix[:, :, max_z] = 1
    # enclosing_coords = make_coordinates(enclosing_matrix)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d', proj_type='ortho')
    # ax.scatter(coords[::5, 0], coords[::5, 1], coords[::5, 2], color='green', alpha=1)
    # ax.scatter(slice_coords[::20, 0], slice_coords[::20, 1], slice_coords[::20, 2], color='blue', alpha=0.5)
    # ax.scatter(whole[::20, 0], whole[::20, 1], whole[::20, 2], color='yellow', alpha=0.5)
    # ax.scatter(outer_coords[::20, 0], outer_coords[::20, 1], outer_coords[::20, 2], color='red', alpha=0.3)

    # ax.scatter(enclosing_coords[::, 0], enclosing_coords[::, 1], enclosing_coords[::, 2], color='red', alpha=0.1)

    plt.show()


if __name__ == '__main__':
    main()
    # test_bin_stat()
# Execution time of func "calculate_density": 156.1464276999468 s
# Execution time of func "calculate_density": 112.50546269991901 s
# Execution time of func "calculate_density": 108.97752479999326 s
