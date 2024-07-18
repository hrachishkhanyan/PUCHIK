from time import perf_counter

from PUCHIK import Interface


def test_sphere():
    sphere_pdb = 'test_structures/InP_sphere_r_29.pdb'
    selection = f'resname UNL or resname SOL and not type H'

    mesh = Interface(sphere_pdb)
    mesh.select_atoms(selection)

    vol = mesh.calculate_volume()
    print(f'Expected volume: 102.160404\nCalculated volume: {vol}')


def test_cylinder():
    sphere_pdb = 'test_structures/InP_cylinder.pdb'
    selection = f'resname UNL or resname SOL and not type H'

    mesh = Interface(sphere_pdb)
    mesh.select_atoms(selection)

    vol = mesh.calculate_volume()
    print(f'Expected volume: 153.24\nCalculated volume: {vol}')


def test_stretch_cyl():
    sphere_pdb = 'test_structures/InP_cylinder.pdb'
    selection = f'resname UNL or resname SOL and not type H'

    mesh = Interface(sphere_pdb)
    mesh.select_atoms(selection)
    vol = mesh.calculate_volume()
    print(f'Volume: {vol}')


if __name__ == '__main__':
    # test_cylinder()
    # test_sphere()
    ...
