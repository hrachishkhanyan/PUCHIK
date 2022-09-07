import functools
import operator
import sys
from collections import Counter
from MDAnalysis.analysis.distances import self_distance_array
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm

# Local imports
from utilities.decorators import timer

np.set_printoptions(threshold=sys.maxsize)
np.seterr(invalid='ignore', divide='ignore')

"""
    Grid method for analyzing complex shaped structures 
"""

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
        self.length = len(self.u.trajectory)
        self.unique_resnames = None

    def select_atoms(self, sel):
        """
        Method for selecting the atoms using MDAnalysis selections

        Args:
            sel (str): selection string

        """
        self.ag = self.u.select_atoms(sel)
        self.unique_resnames = np.unique(self.ag.resnames)
        print(f'Selected atom group: {self.ag}')

    def _get_int_dim(self):
        """
        Utility function to get box dimensions

        Returns:
            Dimensions of the box as an int
        """
        return int(np.ceil(self.dim[0]))

    @staticmethod
    def _generate_point(dim):  # TODO! Move this and else related to MC to somewhere else
        """
        Generates a 3D point within dim boundaries
        Args:
            dim (int): Boundaries of the coordinates

        Returns:
            point (np.ndarray): Generated point
        """
        point = np.random.rand(3) * dim
        return point

    def _monte_carlo(self, dim, number):
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
        for _ in range(number):
            point = self._generate_point(dim)
            which_cell = self.check_cube(*point)
            if self.mesh[which_cell] == 1:
                in_volume += 1
            else:
                out_volume += 1

        ratio = in_volume / (out_volume + in_volume)
        return ratio

    def _monte_carlo_volume(self, number, rescale=None):
        """
        Utility function responsible for rescaling and calling the actual algorithm

        Args:
            number (int): Number of points to generate
            rescale (int): Rescale factor

        Returns:
            float: Estimated volume of the structure
        """
        rescale_coef = self.find_min_dist() if rescale is None else rescale
        # pbc_vol = (self._get_int_dim() // rescale_coef) ** 3
        pbc_vol = self._get_int_dim() ** 3

        pbc_sys_ratio = self._monte_carlo(self._get_int_dim() // rescale_coef, number=number)

        return pbc_sys_ratio * pbc_vol

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

        vol = self._monte_carlo_volume(number, rescale) if method == 'mc' else None

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

    @staticmethod
    def _is_inside(point, mesh):
        dep, row, col = point

        x_mesh_indices = np.where(mesh[:, row, col] > 0)[0]
        y_mesh_indices = np.where(mesh[dep, :, col] > 0)[0]
        z_mesh_indices = np.where(mesh[dep, row, :] > 0)[0]
        # print(point, x_mesh_indices, y_mesh_indices, z_mesh_indices)
        if len(x_mesh_indices) == 0 or len(y_mesh_indices) == 0 or len(z_mesh_indices) == 0:
            return False

        dep_mesh_min, dep_mesh_max = np.where(mesh[:, row, col] > 0)[0].min(), np.where(mesh[:, row, col] > 0)[0].max()
        row_mesh_min, row_mesh_max = np.where(mesh[dep, :, col] > 0)[0].min(), np.where(mesh[dep, :, col] > 0)[0].max()
        col_mesh_min, col_mesh_max = np.where(mesh[dep, row, :] > 0)[0].min(), np.where(mesh[dep, row, :] > 0)[0].max()

        if (row_mesh_min <= row <= row_mesh_max
                and col_mesh_min <= col <= col_mesh_max
                and dep_mesh_min <= dep <= dep_mesh_max):
            return True

        return False

    def _calc_mesh(self, grid_dim, rescale):
        """
        Calculates the mesh according the atom positions in the box

        Args:
            grid_dim (int): Box dimensions
            rescale: rescale factor

        Returns:
            np.ndarray: The grid
        """
        grid_matrix = self.make_grid(grid_dim, dim=rescale, d4=len(self.unique_resnames))

        for atom in self.ag:
            x, y, z = self.check_cube(*atom.position, rescale=rescale)
            res_number = np.where(self.unique_resnames == atom.resname)
            grid_matrix[x, y, z, res_number] += 1

        return grid_matrix

    def _calc_density(self, mol_type, grid_dim, min_distance_coeff):
        """ Not sure what's this for. May delete it later """
        density_matrix = self.make_grid(grid_dim, dim=min_distance_coeff, d4=False)
        for atom in self.ag:
            # print(atom.position)
            x, y, z = self.check_cube(*atom.position, rescale=min_distance_coeff)
            if atom.type == mol_type:
                density_matrix[x, y, z] += 1

        return density_matrix

    @timer
    def calculate_mesh(self, rescale=None):
        """
        Calculates the mesh using _calc_mesh private method
        Args:
            rescale: rescale factor

        Returns:
            np.ndarray: Returns the grid matrix
        """
        # find closest atoms and rescale positions according to this
        min_distance_coeff = self.find_min_dist() if rescale is None else rescale
        grid_dim = self._get_int_dim()  # get one dimension

        # define the matrices
        grid_matrix = self._calc_mesh(grid_dim, min_distance_coeff)
        self.grid_matrix = grid_matrix
        # density_matrix = self._calc_density('IN', grid_dim, min_distance_coeff)
        return grid_matrix
        # grid_matrix = np.asarray(grid_matrix[:, :, :, 1] > grid_matrix[:, :, :, 0], dtype=int)
        # self.mesh = grid_matrix
        # return grid_matrix, density_matrix

    @timer
    def calculate_interface(self, inverse=False):  # FIXME: This should use the whole structure
        """
        Extract the interface from the grid TODO better way needed
        Args:
            inverse (bool): Return everything except for the structure if True

        Returns:
            np.ndarray: interface matrix
        """
        # interface = np.asarray(self.grid_matrix[:, :, :, 1] / self.grid_matrix[:, :, :, 0] > 0.6, dtype=int)
        interface = self.grid_matrix.copy()

        if inverse:
            interface[self.grid_matrix[:, :, :, 1] / self.grid_matrix[:, :, :, 0] >= 0.4] = 0
            return interface[:, :, :, 0]

        interface[(0 < self.grid_matrix[:, :, :, 1] / self.grid_matrix[:, :, :, 0]) & (
                self.grid_matrix[:, :, :, 1] / self.grid_matrix[:, :, :, 0] < 0.4)] = 0

        interface = interface[:, :, :, 1]
        return interface

    @timer
    def _find_distance(self, points, mesh_coords, mesh):
        """
        Finds the distances between points and mesh coordinates
        :param points (ndarray): Coordinates of points that is an ndarray containing 3D coordinates
        :param mesh_coords (ndarray): Mesh coordinates. Is an ndarray containing 3D coordinates
        :param mesh (ndarray): Mesh in 3D matrix form. Needed to quickly determine whether the point is inside structure
        or outside
        :return tuple: length 2 tuple. First value is the number of particles in each node. Second is a dictionary with
        keys as distances and values as lists of nodes at that distance
        """

        dists = []
        dists_dict = {}
        node_count = []

        for i, point in enumerate(points):
            min_dist = 10  # some starting distance. FIXME take from the existing ones
            coord, num = point[0:3], point[3]  # num is the number of particles at node coord
            inside = Mesh._is_inside(coord, mesh)  # flag to determine if the point is inside the mesh

            for mesh_point in mesh_coords:

                dist = norm(mesh_point - coord)

                if dist < min_dist:
                    min_dist = dist

            if inside:
                min_dist *= -1  # distance from interface is negative if inside

            temp = [min_dist]
            node_count += temp

            if not dists_dict.get(min_dist):
                dists_dict[min_dist] = []

            dists_dict[min_dist] += (temp * num)
            dists += (temp * num)

        return node_count, dists_dict

    def _normalize_density(self, node_count, dist_dict):

        # Normalization TODO Write a separate function
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

    @timer
    def calculate_density(self, selection=None, skip=1):
        """
        Calculates the density of selection from interface

        Args:
            selection (str): Selection of the atom group density of which is to be calculated
            skip (int): Use every 'skip'-th frame to calculate density
        Returns:
            tuple: Density array and corresponding distances
        """
        frame_count = self.length // skip
        res = [None] * (frame_count + 1)
        dists_dict_list = [{}] * (frame_count + 1)
        index = 0  # index of the frame

        for j, ts in enumerate(self.u.trajectory):
            if j % skip == 0:
                self.calculate_mesh(rescale=self.rescale)
                interface = self.calculate_interface()
                mesh_coordinates = self.make_coordinates(interface)
                # inverse = self.calculate_interface(inverse=True)
                # points = self.make_coordinates(inverse, keep_numbers=True)
                inverse = self._extract_from_mesh(selection)
                points = self.make_coordinates(inverse, keep_numbers=True)

                res[index], d = self._find_distance(points, mesh_coordinates, interface)
                dists_dict_list[index] = self._normalize_density(res[index], d)
                index += 1

        res = dict(functools.reduce(operator.add, map(Counter, dists_dict_list)))
        # Average over time
        res = {k: res[k] for k in sorted(res)}
        res = np.array(list(res.keys())), np.array(list(res.values())) / (frame_count + 1)
        # res = np.array(res).mean(axis=0)
        return res  # Return densities and according distances

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


def main():
    tx_100 = r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_7\100pc_tyl\production.part0005.gro'
    selection = f'resname TY79 TX0 TIP3 and not type H'
    rescale = 5

    mesh = Mesh(traj=r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_7\75tyl_25TX\centered.xtc',
                top=r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_7\75tyl_25TX\centered.gro', rescale=rescale)

    mesh.select_atoms(selection)
    grid_matrix = mesh.calculate_mesh(rescale=rescale)
    np.save('test/grid.npy', grid_matrix[0])
    d, dens = mesh.calculate_density('TIP3', skip=100)
    d_1, dens_1 = mesh.calculate_density('TY79', skip=100)
    d_2, dens_2 = mesh.calculate_density('TX0', skip=100)

    # coords = mesh.make_coordinates(tx_0)
    # coords_2 = mesh.make_coordinates(ty_39)
    # coords_3 = mesh.make_coordinates(tip3)

    # surf_matrix = np.asarray(grid_matrix[:, :, :, 1] > grid_matrix[:, :, :, 0], dtype=int)
    # water_matrix = np.asarray(grid_matrix[:, :, :, 1] < grid_matrix[:, :, :, 0], dtype=int)

    # int_coords = mesh.make_coordinates(interface)
    # interface = mesh.calculate_interface()
    # inverse = mesh.calculate_interface(inverse=True)
    # print(inverse[0].mean())
    # coords_num = mesh.make_coordinates(interface, keep_numbers=True)
    # interface_1 = mesh.interface(interface)
    # int_coords = mesh.make_coordinates(interface)
    # int_coords_1 = mesh.make_coordinates(interface_1)
    # inverse_coords = mesh.make_coordinates(inverse, keep_numbers=True)
    # for i in range(120 // rescale):
    #     np.save(f'../files/interface_test_files/frame_{i}.npy', interface[:, :, i])
    #     sns.heatmap(interface[:, :, i])
    #     plt.show()
    # time.sleep(.1)
    # print(inverse_coords.shape)
    # print(int_coords.shape)
    # coords = mesh.make_coordinates(surf_matrix)
    #
    # coords_2 = mesh.make_coordinates(water_matrix)

    # d, dens = mesh.calculate_density()
    #
    plt.plot(d, dens)
    plt.plot(d_1, dens_1)
    plt.plot(d_2, dens_2)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d', proj_type='ortho')
    # ax.scatter(int_coords[:, 0], int_coords[:, 1], int_coords[:, 2], color='red')
    # ax.scatter(inverse_coords[:, 0], inverse_coords[:, 1], inverse_coords[:, 2], color='blue', linewidth=0.2)

    # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='green', alpha=0.5)
    # ax.scatter(coords_2[:, 0], coords_2[:, 1], coords_2[:, 2], color='red')
    # ax.scatter(coords_3[:, 0], coords_3[:, 1], coords_3[:, 2], color='blue')

    # ax.grid(False)

    plt.show()


if __name__ == '__main__':
    main()
    # test_bin_stat()
