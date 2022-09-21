import setuptools
from setuptools import setup

setup(name='Grid Project',
      version='1.0.0',
      description='Intrinsic density profiles for aspherical structures',
      url='https://github.com/hrachishkhanyan/grid_project',
      author='H. Ishkhanyan',
      author_email='hrachya.ishkhanyan@kcl.ac.uk',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[
          'MDAnalysis',
      ])
