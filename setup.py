from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

VERSION = '1.0.1'

extensions = [
    Extension(
        name='AICON.grid_project.core.utils',
        sources=['AICON/grid_project/core/utils.pyx'],
        include_dirs=[numpy.get_include(), 'AICON/grid_project/core'],
    )
]

setup(
    name='AICON',
    version=VERSION,
    description='Intrinsic density profiles for aspherical structures',
    url='https://github.com/hrachishkhanyan/grid_project',
    author='H. Ishkhanyan',
    author_email='hrachya.ishkhanyan@kcl.ac.uk',
    license='MIT',
    provides=['AICON'],
    ext_modules=cythonize(extensions),
)
