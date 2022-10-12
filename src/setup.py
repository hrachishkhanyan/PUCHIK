from setuptools import setup, find_packages


setup(name='Grid Project',
      version='1.0.7',
      description='Intrinsic density profiles for aspherical structures',
      url='https://github.com/hrachishkhanyan/grid_project',
      author='H. Ishkhanyan',
      author_email='hrachya.ishkhanyan@kcl.ac.uk',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'MDAnalysis',
      ])
