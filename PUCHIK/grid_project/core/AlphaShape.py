import subprocess
import numpy as np
from dotenv import load_dotenv
import os

# Load .env to get alpha_shape exe
load_dotenv()


class AlphaShape:

    def __init__(self, points):
        self._points = points  # All points
        self._cells = None
        self._simplices = None

    @property
    def cells(self):
        return self._points

    @cells.setter
    def cells(self, new_cells):
        self._cells = new_cells

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, new_points):
        self._simplices = new_points

    @property
    def simplices(self):
        return self._simplices

    @simplices.setter
    def simplices(self, new_simplices):
        self._simplices = new_simplices

    def calculate_as(self, alpha=-1):
        alpha_shaper_exe = os.getenv('ALPHA_SHAPER_EXECUTABLE')

        np.savetxt('.temp.xyz', self.points, header=f'{len(self.points)}', comments='')
        proc = subprocess.run([alpha_shaper_exe, './.temp.xyz', f'{alpha}'], capture_output=True, text=True)
        os.remove('.temp.xyz')

        self.simplices = np.loadtxt('output_facets.txt', dtype=int)
        self.cells = np.loadtxt('output_cells.txt', dtype=int)
        print(proc.stdout)
        return self
