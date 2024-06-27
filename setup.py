import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Import the current version number
# from AICON._version import __version__
import tomllib


def get_current_version():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    return data['project']['version']


extensions = [
    Extension(
        name='AICON.grid_project.core.utils',
        sources=['AICON/grid_project/core/utils.pyx'],
        include_dirs=[numpy.get_include(), 'AICON/grid_project/core'],
    )
]

setup(
    name='AICON',
    version=get_current_version(),
    description='Intrinsic density profiles for aspherical structures',
    url='https://github.com/hrachishkhanyan/grid_project',
    author='H. Ishkhanyan',
    author_email='hrachya.ishkhanyan@kcl.ac.uk',
    license='MIT',
    provides=['AICON'],
    ext_modules=cythonize(extensions),
)
