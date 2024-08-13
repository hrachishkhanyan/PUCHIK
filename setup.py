import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Import the current version number
# from PUCHIK._version import __version__


extensions = [
    Extension(
        name='PUCHIK.grid_project.core.utils',
        sources=['PUCHIK/grid_project/core/utils.pyx'],
        include_dirs=[numpy.get_include(), 'PUCHIK/grid_project/core'],
    )
]

setup(
    name='PUCHIK',
    version='1.1.1',
    description='Python Utility for Characterizing Heterogeneous Interfaces and Kinetics',
    url='https://github.com/hrachishkhanyan/grid_project',
    author='H. Ishkhanyan',
    author_email='hrachya.ishkhanyan@kcl.ac.uk',
    license='MIT',
    provides=['PUCHIK'],
    ext_modules=cythonize(extensions),
)
