import subprocess
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env to get alpha_shape exe
load_dotenv()


class AlphaShape:
    """ Python wrapper for AlphaShaper.exe """
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

    def calculate_as(self, frame_num, alpha=-1):
        """
        Calculate the alpha shape for frame <frame_num>
        :param frame_num:
        :param alpha:
        :return:
        """
        temp_file_name = f'./.temp_frame_{frame_num}.xyz'
        temp_output_facets_file_name = f'output_facets_{frame_num}.txt'
        temp_output_cells_file_name = f'output_cells_{frame_num}.txt'

        output_file_suffix = f'{frame_num}'
        alpha_shaper_exe = Path(__file__).resolve().parent.parent / 'alpha_shaper' / 'AlphaShaper.exe'
        if not os.path.exists(alpha_shaper_exe):
            print(alpha_shaper_exe)
            raise FileNotFoundError("Couldn't find the executable.")

        np.savetxt(temp_file_name, self.points, header=f'{len(self.points)}', comments='')
        proc = subprocess.run([alpha_shaper_exe, temp_file_name, f'{alpha}', output_file_suffix],
                              capture_output=True, text=True)

        self.simplices = np.loadtxt(temp_output_facets_file_name, dtype=int)
        self.cells = np.loadtxt(temp_output_cells_file_name, dtype=int)

        os.remove(temp_file_name)
        os.remove(temp_output_facets_file_name)
        os.remove(temp_output_cells_file_name)

        return self
