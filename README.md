# PUCHIK

## Overview
Python Utility for Characterizing Heterogeneous Interfaces and Kinetics (PUCHIK), is a tool for analyzing molecular dynamics trajectories. It allows constructing an interface between two phases, enabling to calculate intrinsic density profiles, volumes, etc.

The interface construction works for spherical and rod-like nanoparticles equally well, making it a great tool to work with nanoparticles of almost every shape. 

This package is built on top of [MDAnalysis](https://www.mdanalysis.org/), [SciPy](https://scipy.org/), [NumPy](https://numpy.org/doc/stable/index.html) and [PyGEL3D](https://pypi.org/project/PyGEL3D/) libraries.

![image](https://drive.google.com/uc?export=view&id=1YTiM2OxzkGO0GcbC5WvFffBdZN9-e_6D)

## Installation

You can install the PUCHIK package using pip:

```
pip install PUCHIK
```

## Usage

The main class in this package is the "Mesh" class. To set up a mesh, import it from PUCHIK:

```python
from PUCHIK import Mesh
```

You should provide it with a topology and optionally a trajectory files. PUCHIK uses MDAnalysis Readers to open a trajectory. You can find the supported formats [here](https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html).

```python
trj = '<path_to_trajectory>'
top = '<path_to_topology>'
m = Mesh(trj, top)
```

Lastly, select the atom groups you want to consider, atom groups that comprise the interface, and run the **calculate_density** method:

```python
m.select_atoms('all')  # Consider every atom in the system
m.select_structure('<selection>')  # resname of the nanoparticle

density_selection = 'resname TIP3'
m.calculate_density(density_selection)
```

Note that **calculate_density** uses every CPU core. You can specify the number of cores you want to use with the keyword argument **cpu_count**.

A more customized usage of the **calculate_density** method can be:

```python
m.calculate_density(density_selection, start=10, end=1000, skip=2, norm_bin_count=12)
```

This version will start the calculation at the 10th frame and finish it at frame 1000 considering every 2nd frame. **norm_bin_count** specifies the number of divisions of the simulation box in each dimension to create a grid.

An example figure which shows the number density of different residues relative to the distance to the interface of a sodium oleate micelle:

![image](https://drive.google.com/uc?export=view&id=1swRuoD-rs01SA-4jqPLjwxuuv9UlUGkm)