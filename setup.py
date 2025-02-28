import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Import the current version number
# from PUCHIK._version import __version__
# import tomllib  # no tomllib before 3.11


def get_current_version():
    return "1.2.3"
#     with open("pyproject.toml", "rb") as f:
#         data = tomllib.load(f)
#
#     return data['project']['version']


extensions = [
    Extension(
        name='PUCHIK.grid_project.core.utils',
        sources=['PUCHIK/grid_project/core/utils.pyx'],
        include_dirs=[numpy.get_include(), 'PUCHIK/grid_project/core'],
    )
]

setup(
    name='PUCHIK',
    version=get_current_version(),
    description='Python Utility for Characterizing Heterogeneous Interfaces and Kinetics',
    url='https://github.com/hrachishkhanyan/grid_project',
    author='H. Ishkhanyan',
    author_email='hrachya.ishkhanyan@kcl.ac.uk',
    license='MIT',
    provides=['PUCHIK'],
    packages=['PUCHIK'],
    ext_modules=cythonize(extensions),
)
