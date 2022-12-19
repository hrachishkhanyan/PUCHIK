from functools import partial, reduce
from multiprocessing import Pool, cpu_count
import operator
from collections import Counter
from MDAnalysis.analysis.distances import self_distance_array
from MDAnalysis.transformations.wrap import wrap
import MDAnalysis as mda
import numpy as np
from numpy.linalg import norm

# Local imports
from grid_project.utilities.decorators import timer, logger
from grid_project.volume.monte_carlo import monte_carlo_volume
from grid_project.settings import DEBUG
from grid_project.utilities.universal_functions import extract_hull  #, _is_inside

from grid_project.core.utils import find_distance

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
        self.interface_rescale = 4  # this is for calculating a rescaled interface then upscaling it
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

    # @staticmethod
    # def _is_inside(point, mesh):
    #     """ Not a good version. Use the one from utilities.universal_functions """
    #     dep, row, col = np.asarray(point, dtype=int)
    #
    #     x_mesh_indices = np.where(mesh[:, row, col] > 0)[0]
    #     y_mesh_indices = np.where(mesh[dep, :, col] > 0)[0]
    #     z_mesh_indices = np.where(mesh[dep, row, :] > 0)[0]
    #     # print(point, x_mesh_indices, y_mesh_indices, z_mesh_indices)
    #     if len(x_mesh_indices) == 0 or len(y_mesh_indices) == 0 or len(z_mesh_indices) == 0:
    #         return False
    #
    #     dep_mesh_min, dep_mesh_max = np.where(mesh[:, row, col] > 0)[0].min(), np.where(mesh[:, row, col] > 0)[0].max()
    #     row_mesh_min, row_mesh_max = np.where(mesh[dep, :, col] > 0)[0].min(), np.where(mesh[dep, :, col] > 0)[0].max()
    #     col_mesh_min, col_mesh_max = np.where(mesh[dep, row, :] > 0)[0].min(), np.where(mesh[dep, row, :] > 0)[0].max()
    #
    #     if (row_mesh_min <= row <= row_mesh_max
    #             and col_mesh_min <= col <= col_mesh_max
    #             and dep_mesh_min <= dep <= dep_mesh_max):
    #         return True
    #
    #     return False

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
        interface_hull = extract_hull(interface)
        transposed = extract_hull(interface.T).T  # This is done for filling gaps in the other side
        interface_hull += transposed
        return interface_hull

    # @logger(DEBUG)
    # def _find_distance(self, points, mesh_coords, mesh):
    #     """
    #     Finds the distances between points and mesh coordinates
    #     :param points (ndarray): Coordinates of points that is an ndarray containing 3D coordinates
    #     :param mesh_coords (ndarray): Mesh coordinates. Is an ndarray containing 3D coordinates
    #     :param mesh (ndarray): Mesh in 3D matrix form. Needed to quickly determine whether the point is inside structure
    #     or outside
    #     :return tuple: length 2 tuple. First value is the number of particles in each node. Second is a dictionary with
    #     keys as distances and values as lists of nodes at that distance
    #     """
    #     # TODO! Move irrelevant pieces to a separate function
    #     # dists = []
    #     dists_dict = {}
    #     node_count = []
    #
    #     for i, point in enumerate(points):
    #         min_dist = 1000  # some starting distance. FIXME take from the existing ones
    #         coord = point[0:3]
    #         num = point[3]  # num is the number of particles at node coord
    #         inside = _is_inside(coord, mesh)  # flag to determine if the point is inside the mesh
    #
    #         for mesh_point in mesh_coords:
    #             dist = norm(mesh_point - coord)
    #
    #             if dist < min_dist:
    #                 min_dist = dist
    #         if inside:
    #             min_dist *= -1  # distance from interface is negative if inside
    #
    #         temp = [min_dist]
    #         node_count += temp
    #
    #         if not dists_dict.get(min_dist):  # Check if the distance is in the dictionary
    #             dists_dict[min_dist] = []  # create it if not
    #
    #         dists_dict[min_dist] += (temp * num)
    #         # dists += (temp * num)
    #
    #     return node_count, dists_dict

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

    def _extract_from_mesh(self, mol_type):
        if mol_type not in self.unique_resnames:
            raise ValueError(
                f'Molecule type "{mol_type}" is not present in the system. Available types: {self.unique_resnames}'
            )

        mol_index = np.where(self.unique_resnames == mol_type)

        return self.grid_matrix[:, :, :, mol_index]

    def calculate_density(self, interface_selection, selection=None, skip=1, number_of_frames=None):
        """
        Calculates the density of selection from interface

        Args:
            selection (str): Selection of the atom group density of which is to be calculated
            skip (int): Use every 'skip'-th frame to calculate density
            interface_selection (str):
            number_of_frames: For profiling only
        Returns:
            tuple: Density array and corresponding distances

        """

        frame_count = self.length // skip if number_of_frames is None else number_of_frames
        res = [None] * frame_count
        # dists_dict_list = [{}] * (frame_count + 1)
        index = 0  # index of the frame
        loop_range = range(0, frame_count) if number_of_frames is None else np.linspace(0, self.length,
                                                                                        number_of_frames,
                                                                                        dtype=int,
                                                                                        endpoint=False)

        for i in loop_range:
            frame_num = i * skip if number_of_frames is None else i
            self.u.trajectory[frame_num]
            self.calculate_mesh(selection=interface_selection, main_structure=True, rescale=self.interface_rescale)
            interface = self.calculate_interface()

            mesh_coordinates = self.make_coordinates(interface)

            selection_mesh = self.calculate_mesh(selection, rescale=self.rescale)[:, :, :, 0]
            selection_coords = self.make_coordinates(selection_mesh, keep_numbers=True)
            np.save(r'C:\Users\hrach\PycharmProjects\cython_playground\mesh_data.npy', np.array([selection_coords, mesh_coordinates, interface]))

            res[index] = find_distance(selection_coords, mesh_coordinates, interface)

            index += 1
        # This can be a general function
        densities = [elem[0] for elem in res]
        dists = [elem[1] for elem in res]
        dists_dict_list = [self._normalize_density(densities[index], d) for index, d in enumerate(dists)]

        res = dict(reduce(operator.add, map(Counter, dists_dict_list)))

        res = {k: res[k] for k in sorted(res)}
        res = np.array(list(res.keys())), np.array(list(res.values())) / len(loop_range)
        # res = np.array(res).mean(axis=0)
        return res  # Return densities and according distances

    def _calc_dens_mp(self, frame_num, selection, interface_selection, ratio):
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

        self.calculate_mesh(selection=interface_selection, main_structure=True, rescale=self.interface_rescale)
        # interface = stretch(self.calculate_interface(ratio=ratio), self.interface_rescale, 3)  # uncomment after
        # implementing generalized normalization
        interface = self.calculate_interface(ratio=ratio)
        mesh_coordinates = self.make_coordinates(interface)
        # inverse = self.calculate_mesh(selection, rescale=self.rescale)[:, :, :, 0]

        selection_mesh = self.calculate_mesh(selection, rescale=self.rescale)[:, :, :, 0]

        selection_coords = self.make_coordinates(selection_mesh, keep_numbers=True)

        res, d = find_distance(selection_coords, mesh_coordinates, interface)

        return res, d  # Return density and according distance

    # @timer
    def calculate_density_mp(self, selection=None, interface_selection=None, ratio=0.4, start=0, skip=1,
                             cpu_count=CPU_COUNT):
        """
        Calculates density of selection from the interface
        :param selection: MDAnalysis selection of ag
        :param interface_selection: Selection of what is considered as interface
        :param ratio:
        :param start: Starting frame
        :param skip: Skip every n-th frame
        :return:
        """
        n_frames = self.u.trajectory.n_frames

        dens_per_frame = partial(self._calc_dens_mp,
                                 selection=selection,
                                 interface_selection=interface_selection,
                                 ratio=ratio)  # _calc_dens_mp function with filled selection using partial
        frame_range = range(start, n_frames, skip)

        with Pool(cpu_count) as worker_pool:
            res = worker_pool.map(dens_per_frame, frame_range)

        # return res
        densities = [elem[0] for elem in res]
        dists = [elem[1] for elem in res]
        dists_dict_list = [self._normalize_density(densities[index], d) for index, d in enumerate(dists)]

        res = dict(reduce(operator.add, map(Counter, dists_dict_list)))

        res = {k: res[k] for k in sorted(res)}
        res = np.array(list(res.keys())), np.array(list(res.values())) / len(frame_range)
        return res

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
    from src.test.mol_parts import TYL3_HYDROPHOBIC, TX100_HYDROPHOBIC
    from grid_project.settings import BASE_DATA_SAVE_DIR
    TRITO_HYDROPHOBIC = 'C19 C20 C21 C22 C23 C24 C25 C26 C27 C28 C29 C30 C31'

    paths = {
        'tx100': r'C:\Users\hrach\Documents\Simulations\TX100',
        'mix': r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3'
    }
    rescale = 4
    skip = 2000
    system = ['TY39']
    # interface_selection = f'resname TRITO and name {TRITO_HYDROPHOBIC} or resname TX0 and not name {TX100_HYDROPHOBIC} ' \
    #                       f'and not type H'
    interface_selection = f'resname TY39 and name {TYL3_HYDROPHOBIC} and not type H'
    surf_ratio = 'NA'
    ratio = 0.6
    mesh = Mesh(
        traj=fr'{paths["mix"]}\100pc_tyl\centered_whole_skip_10.xtc',
        top=fr'{paths["mix"]}\100pc_tyl\centered.gro',
        rescale=rescale)
    mesh.interface_rescale = rescale
    mesh.select_atoms('not type H')
    mesh.select_structure(*system)

    d, dens = mesh.calculate_density(selection='resname TIP3 and not type H',
                                     interface_selection=interface_selection, skip=skip, number_of_frames=1)

    # d_1, dens_1 = mesh.calculate_density_mp(
    #     f'resname TY39 and name {TYL3_HYDROPHOBIC} and not type H O', interface_selection=interface_selection,
    #     ratio=ratio,
    #     skip=skip)  # hydrophobic
    # d_2, dens_2 = mesh.calculate_density_mp(
    #     f'resname TY39 and not name {TYL3_HYDROPHOBIC} and not type H O', interface_selection=interface_selection,
    #     ratio=ratio,
    #     skip=skip)  # hydrophilic
    # d_3, dens_3 = mesh.calculate_density_mp(
    #     f'resname TX0 and not name {TX100_HYDROPHOBIC} and not type H O', interface_selection=interface_selection,
    #     ratio=ratio,
    #     skip=skip)  # hydrophobic
    # d_4, dens_4 = mesh.calculate_density_mp(
    #     f'resname TX0 and name {TX100_HYDROPHOBIC} and not type H O', interface_selection=interface_selection,
    #     ratio=ratio,
    #     skip=skip)  # hydrophilic

    # np.save(
    #     f'{BASE_DATA_SAVE_DIR}/{"_".join(system)}_{surf_ratio}_data_{rescale}_rescaled_{str(ratio).replace("․", "_")}_with_Oxy.npy',
    #     np.array([d, dens, d_1, dens_1, d_2, dens_2], dtype=object))  # , d_3, dens_3, d_4, dens_4], dtype=object))
    # d, dens, d_1, dens_1, d_2, dens_2, d_3, dens_3, d_4, dens_4 = np.load(
    #     f'{DATA_DIR}/{"_".join(system)}_{surf_ratio}_data_{rescale}_rescaled.npy', allow_pickle=True)
    # plt.hist([i[0] for i in res[1].values()])

    from matplotlib import pyplot as plt
    plt.plot(d, dens)
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
