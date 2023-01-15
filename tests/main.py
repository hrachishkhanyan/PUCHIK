import operator
from collections import Counter
from functools import reduce

import numpy as np
from MDAnalysis.analysis.rdf import InterRDF
from grid_project.utilities.universal_functions import make_coordinates
from matplotlib import pyplot as plt
import pickle
from grid_project import Mesh
from scipy.spatial import ConvexHull

from mol_parts import TYL3_HYDROPHOBIC, TX100_HYDROPHOBIC

DATA_DIR = r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\data\\'


def main(*args, **kwargs):
    rescale = 4
    skip = 1000
    system = ['TY39', 'TX0']
    surf_ratio = '50_50'
    ratio = 0.4
    mesh = Mesh(
        traj=r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\50tyl_50TX\production.part0005_skip_10.xtc',
        top=r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\50tyl_50TX\production.part0005.gro',
        rescale=rescale)
    mesh.interface_rescale = rescale
    # mesh = Mesh(r'C:\Users\hrach\Documents\Simulations\TX100\triton_micelle_production999.gro', rescale=rescale)
    # tail = 'C6A C6B C6C C5 C4 C2 C3A C3B C1A C1B C1F C1C C1E C1D'
    # head = 'O1 CA1 CA2 O2 CB1 CB2 O3 CC1 CC2 O4 CD1 CD2 O5 CE1 CE2 O6 CF1 CF2 O7 CG1 CG2 O8 CJ1 CJ2 O11'
    tail = 'C'
    mesh.select_atoms('not type H')
    mesh.select_structure(*system)
    # grid_matrix = mesh.calculate_mesh(rescale=rescale)
    # interface = mesh.calculate_interface()
    # min_z, max_z, min_y, max_y, min_x, max_x = mesh.interface_borders
    # s = mesh.u.select_atoms(
    #     f'not type H and (prop x > {min_x} and prop x < {max_x}) and (prop y > {min_y} and prop y < {max_y})'
    #     f' and (prop z > {min_z} and prop z < {max_z})')
    # s_1 = mesh.u.select_atoms('all')
    # coords = mesh.make_coordinates(interface)
    # outer_mesh = mesh.calculate_mesh('resname SOL and not type H', rescale=rescale)[:, :, :, 0]
    # outer_mesh[min_x:max_x, min_y:max_y, min_z:max_z] = 0  # TODO: This too
    # outer_coords = mesh.make_coordinates(outer_mesh)
    # print(outer_coords)
    # slice_coords = s.positions
    # whole = s_1.positions
    # res = mesh._calc_inner_dens('resname SOL and not type H', coords, interface)
    # np.save(r'C:\Users\hrach\Documents\innter_density_test.npy', [i[0] for i in res[1].values()])
    d, dens = mesh.calculate_density(selection='resname TIP3 and not type H',
                                     interface_selection=interface_selection, skip=skip,
                                     number_of_frames=1)  # d_1, dens_1 = mesh.calculate_density_mp(
    #     f'resname TY39 and name {TYL3_HYDROPHOBIC} and not type H O', ratio=ratio,
    #     skip=skip)  # hydrophobic
    # d_2, dens_2 = mesh.calculate_density_mp(
    #     f'resname TY39 and not name {TYL3_HYDROPHOBIC} and not type H O', ratio=ratio,
    #     skip=skip)  # hydrophilic
    # d_3, dens_3 = mesh.calculate_density_mp(
    #     f'resname TX0 and not name {TX100_HYDROPHOBIC} and not type H O', ratio=ratio,
    #     skip=skip)  # hydrophobic
    # d_4, dens_4 = mesh.calculate_density_mp(
    #     f'resname TX0 and name {TX100_HYDROPHOBIC} and not type H O', ratio=ratio,
    #     skip=skip)  # hydrophilic
    np.save(
        f'{DATA_DIR}/water_{"_".join(system)}_{surf_ratio}_data_{rescale}_rescaled_{str(ratio).replace("․", "_")}.npy',
        np.array([d, dens]))  # , d_1, dens_1, d_2, dens_2, d_3, dens_3, d_4, dens_4], dtype=object))
    # d, dens, d_1, dens_1, d_2, dens_2, d_3, dens_3, d_4, dens_4 = np.load(
    #     f'{DATA_DIR}/{"_".join(system)}_{surf_ratio}_data_{rescale}_rescaled.npy', allow_pickle=True)
    # plt.hist([i[0] for i in res[1].values()])

    plt.plot(d, dens)
    # plt.plot(d_1, dens_1, label='TY39 Hydrophobic')
    # plt.plot(d_2, dens_2, label='TY39 Hydrophilic')
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
    # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='green', alpha=1)
    # ax.scatter(slice_coords[::20, 0], slice_coords[::20, 1], slice_coords[::20, 2], color='blue', alpha=0.5)
    # ax.scatter(whole[::20, 0], whole[::20, 1], whole[::20, 2], color='yellow', alpha=0.5)
    # ax.scatter(outer_coords[::20, 0], outer_coords[::20, 1], outer_coords[::20, 2], color='red', alpha=0.3)

    # ax.scatter(enclosing_coords[::, 0], enclosing_coords[::, 1], enclosing_coords[::, 2], color='red', alpha=0.1)

    plt.show()


def normalize_density(arg, bin_count):
    dists_and_coord = [elem[0] for elem in arg]
    dists = [elem[1] for elem in arg]
    n_frames = len(dists_and_coord)
    # dens, dist = _normalize_density(dists_and_coord[0], dists[0])
    dens, dist = _normalize_density_2(dists_and_coord[0], bin_count)
    # coords = mesh
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d', proj_type='ortho')
    # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='green', alpha=.8, s=.2)
    plt.plot(dist, dens)
    # ax.set_xlim(0, 120)
    # ax.set_ylim(0, 120)
    # ax.set_zlim(0, 120)
    # plt.ylim(0, 0.04)
    plt.show()


BOX_DIM = 120


def _normalize_density_2(dists_and_coord, bin_count=12):
    # bin_count - how many equal parts is one dimension divided
    # Տուփը բաժանում ենք N հավասար խորանարդերի։ Հաշվում ենք բոլոր տուփերի միջև դիստանցիաները ու դրանց համապատասխան
    # մասնիկների քանակները։ Մասնիկները քանակը բաժանում ենք խորանարդի ծավալի վրա ու ^ այս դիստանցիաներից յուրաքանչյուրին
    # վերագրում ենք այդ խտությունը

    res_dists = []
    res_densities = []
    number_matrix = np.empty(shape=(BOX_DIM,) * 3 + (2,))
    number_matrix.fill(np.nan)

    bin_size = BOX_DIM // bin_count

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

    min_d, max_d = int(dist.min()), int(dist.max()) + 1  # considering range limits

    # dens_fin = np.zeros((max_d - min_d))
    dens_fin = np.empty(BOX_DIM)  # Distances are indices of the array.
    dist_fin = np.empty(BOX_DIM)
    dens_fin.fill(np.nan)
    dist_fin.fill(np.nan)
    offset = 25  # Array has an offset of 25 to account for negative distances
    print(min_d, max_d)
    for j in range(min_d, max_d):
        indices = np.argwhere(dist == j)
        dens_fin[offset + j] = dens[indices].mean()
        dist_fin[offset + j] = j

    return dens_fin, dist_fin  # np.unique(dist)


def rdf():
    import MDAnalysis as mda
    traj = r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\100pc_tyl\centered_whole_skip_10.xtc'
    top = r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\100pc_tyl\centered.gro'
    u = mda.Universe(top, traj)
    g1 = u.select_atoms('resid 41 and name C54')
    g2 = u.select_atoms('resname TIP3 and not type H')
    rdf = InterRDF(g1, g2)
    rdf.run(100, 4000, 100)
    plt.plot(rdf.results.bins, rdf.results.rdf)
    plt.show()


def _is_inside_old(point, mesh):
    x, y, z = point
    # yz_proj = mesh.sum(axis=0)
    # xz_proj = mesh.sum(axis=1)
    # xy_proj = mesh.sum(axis=2)

    yz_proj = make_coordinates(mesh.sum(axis=0))
    xz_proj = make_coordinates(mesh.sum(axis=1))
    xy_proj = make_coordinates(mesh.sum(axis=2))

    yz_hull = ConvexHull(yz_proj)
    xz_hull = ConvexHull(xz_proj)
    xy_hull = ConvexHull(xy_proj)

    yz_inside = point_in_hull((y, z), yz_hull)
    xz_inside = point_in_hull((x, z), xz_hull)
    xy_inside = point_in_hull((x, y), xy_hull)

    return yz_inside and xz_inside and xy_inside


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def inside_or_outside(points, mesh_coords):
    dists_and_coord = []  # Will contain distance of the point from the interface and from the origin
    inside_coords = []
    outside_coords = []
    hull = ConvexHull(mesh_coords, qhull_options='QJ')
    for i, point in enumerate(points):
        coord = point[0:3]
        # num = point[3]  # num is the number of particles at node coord
        inside = _is_inside(coord, hull)  # flag to determine if the point is inside the mesh
        if inside:
            inside_coords.append(coord)
        else:
            outside_coords.append(coord)

    return np.array(inside_coords), np.array(outside_coords)


def find_distance_2(points, mesh_coords, mesh):
    dists_and_coord = []  # Will contain distance of the point from the interface and from the origin

    for i, point in enumerate(points):
        min_dist = 1000
        coord = point[0:3]
        # num = point[3]  # num is the number of particles at node coord
        inside = _is_inside(coord, mesh)  # flag to determine if the point is inside the mesh

        for mesh_point in mesh_coords:
            dist = norm(mesh_point, coord)

            if dist < min_dist:
                min_dist = dist

        if inside:
            min_dist *= -1

        dists_and_coord.append((min_dist, coord))

    return dists_and_coord


def norm(p_1, p_2):
    x_1 = p_1[0]
    y_1 = p_1[1]
    z_1 = p_1[2]
    x_2 = p_2[0]
    y_2 = p_2[1]
    z_2 = p_2[2]
    return np.sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2) + (z_1 - z_2) * (z_1 - z_2))

# def _is_inside(points, mesh):
#     hull = ConvexHull(mesh)
#     from scipy.spatial import Delaunay
#
#     if not isinstance(mesh, Delaunay):
#         hull = Delaunay(mesh, qhull_options='QJ Qbb Pp')
#     print(hull.find_simplex(points).max())
#     return hull.find_simplex(points) - 100 >= 0


def _is_inside(point, mesh):
    hull = ConvexHull(mesh)
    return point_in_hull(point, hull)

def _is_inside_nope(point, mesh):
    inside = np.array((False, False, False))
    slice_index = [[1, 2], [0, 2], [0, 1]]
    for i in range(3):
        mesh_slice = mesh[mesh[:, i] == point[i]]
        if len(mesh_slice) >= 3:
            hull = ConvexHull(mesh_slice[:, slice_index[i]], qhull_options='QJ')
            inside[i] = point_in_hull(point[slice_index[i]], hull)

    return inside.all()

def find_distance_3(points, mesh_coords):
    hull = ConvexHull(mesh_coords, qhull_options='QJ')
    dists_and_coord = []

    for i, point in enumerate(points):
        min_dist = 1000
        coord = point[0:3]
        # num = point[3]  # num is the number of particles at node coord
        inside = _is_inside(coord, mesh_coords)  # flag to determine if the point is inside the mesh

        for simplex in hull.simplices:
            # p_0, p_1, p_2 = mesh_coords[simplex]
            # # Calculating normal of the simplex
            # perpend = np.cross(p_1 - p_0, p_2 - p_0)
            # normal = perpend / np.linalg.norm(perpend)
            # # calculate distance
            # sign = np.sign(np.dot(coord - p_0, normal))
            dist = norm(coord, mesh_coords[simplex].mean(axis=0))
            if dist < min_dist:
                min_dist = dist
            sign = -1 if inside else 1

        dists_and_coord.append((sign * min_dist, coord))

    return dists_and_coord

def test_find_distance_3():
    selection_coords = np.load('selection_coords_1.npy')
    mesh_coordinates = np.load('mesh_coordinates_1.npy')
    # data = find_distance_3(selection_coords, mesh_coordinates)
    # with open(r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\data\\new_distances_.pickle',
    #           'wb') as file:
    #     pickle.dump(data, file)
    with open(r'C:\Users\hrach\Documents\Simulations\tyloxapol_tx\tyl_3\data\\new_distances_.pickle', 'rb') as file:
        data = pickle.load(file)

    dens, dist = _normalize_density_2(data)
    print(dens)
    plt.plot(range(-25, 95), dens)
    plt.show()
def test_is_inside():
    selection_coords = np.load('C:\\Users\\hrach\\Documents\\Simulations\\TX100\\grid_project_test\\selection_1980.npy')
    mesh_coordinates = np.load('C:\\Users\\hrach\\Documents\\Simulations\\TX100\\grid_project_test\\mesh_1980.npy')
    # interface = np.load('interface_1.npy')

    # inside = selection_coords[classification]
    # outside = selection_coords[~classification]
    # print(inside.shape)
    inside, outside = inside_or_outside(selection_coords, mesh_coordinates)
    hull = ConvexHull(mesh_coordinates, qhull_options='Qa QJ Q6')
    # np.save('inside_4.npy', inside)
    # np.save('outside_4.npy', outside)
    # inside, outside = np.load('inside.npy'), np.load('outside.npy')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')
    for i in hull.simplices:
        plt.plot(mesh_coordinates[i, 0], mesh_coordinates[i, 1], mesh_coordinates[i, 2], 'r-', alpha=0.2)
    ax.scatter(inside[:, 0], inside[:, 1], inside[:, 2], color='green', alpha=1, s=.2)
    ax.scatter(outside[:, 0], outside[:, 1], outside[:, 2], color='blue', alpha=1, s=.2)

    plt.show()


def func(coords, mesh, hull):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')

    for i in hull.simplices:
        plt.plot(mesh[i, 0], mesh[i, 1], mesh[i, 2], 'r-')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='green', alpha=1, s=.2)
    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], color='red', alpha=1, s=.2)

    plt.show()

if __name__ == '__main__':
    test_is_inside()
    # main()
    # test_find_distance_3()
    # with open(f'{DATA_DIR}/water_rescale_1_new.pickle', 'rb') as file:
    #     o = pickle.load(file)
    # print(o)
    # dens, dist = o
    # dens = np.array(dens)
    # dist = np.array(dist)
    # # plt.plot(dist[2], dens[2])
    # plt.plot(dist.mean(axis=0, where=~np.isnan(dist)), dens.mean(axis=0, where=~np.isnan(dens)))
    # plt.plot(dist.mean(axis=0), dens.mean(axis=0))

    # normalize_density(o, bin_count=20)
    # rdf()
    # mesh = np.load(r'C:\Users\hrach\PycharmProjects\md_grid_project\tests\mesh_test.npy')
    # point = np.array([87, 57, 75])
    # print(_is_inside(point, mesh))
    # yz_proj = make_coordinates(mesh.sum(axis=0))
    # xz_proj = make_coordinates(mesh.sum(axis=1))
    # xy_proj = make_coordinates(mesh.sum(axis=2))
    #
    # plt.scatter(yz_proj[:, 0], yz_proj[:, 1])
    # plt.scatter(point[1], point[2])
    #
    # for simplex in hull.simplices:
    #     plt.plot(xz_proj[simplex, 0], xz_proj[simplex, 1], 'k-')
    # plt.scatter(xz_proj[:, 0], xz_proj[:, 1])
    # plt.scatter(point[0], point[2])
    #
    # plt.scatter(xy_proj[:, 0], xy_proj[:, 1])
    # plt.scatter(point[0], point[1])
    #
    # plt.show()
