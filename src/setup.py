from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(name='grid_project',
      version='1.0.10',
      description='Intrinsic density profiles for aspherical structures',
      url='https://github.com/hrachishkhanyan/grid_project',
      author='H. Ishkhanyan',
      author_email='hrachya.ishkhanyan@kcl.ac.uk',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'cython',
          'MDAnalysis',
          'numpy',
      ],
      ext_modules=cythonize('grid_project/core/utils.pyx'),
      package_dir={'src': ''},
      include_dirs=[numpy.get_include()]
      )
