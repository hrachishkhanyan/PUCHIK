import os
import platform
import subprocess
from pathlib import Path

import numpy as np

# python-dotenv is only needed to pick up an ALPHA_SHAPER_EXECUTABLE override
# from a .env file. It is optional: the package must import fine without it.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class AlphaShape:
    """ Python wrapper for the compiled AlphaShaper executable.

    Alpha-shape support is optional: it requires the CGAL-based AlphaShaper
    binary, which is only bundled for Windows. On any platform where a runnable
    executable cannot be found, :meth:`is_available` returns ``False`` and the
    caller should fall back to convex-hull analysis. """
    def __init__(self, points):
        self._points = points  # All points
        self._cells = None
        self._simplices = None
        self.volume = None

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, new_cells):
        self._cells = new_cells

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, new_points):
        self._points = new_points

    @property
    def simplices(self):
        return self._simplices

    @simplices.setter
    def simplices(self, new_simplices):
        self._simplices = new_simplices

    @staticmethod
    def _resolve_executable():
        """
        Locate a runnable AlphaShaper executable for the current platform.

        Resolution order:
            1. The ``ALPHA_SHAPER_EXECUTABLE`` environment variable (may come
               from a .env file), if it points at a runnable file.
            2. The bundled binary in ``PUCHIK/grid_project/alpha_shaper``, using
               the platform-appropriate name (``AlphaShaper.exe`` on Windows,
               ``AlphaShaper`` elsewhere).

        Returns:
            pathlib.Path | None: Path to a runnable executable, or ``None`` if
            none was found.
        """
        base = Path(__file__).resolve().parent.parent / 'alpha_shaper'
        exe_name = 'AlphaShaper.exe' if platform.system() == 'Windows' else 'AlphaShaper'

        candidates = []
        env_path = os.environ.get('ALPHA_SHAPER_EXECUTABLE')
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(base / exe_name)

        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        return None

    @classmethod
    def is_available(cls):
        """ Return True if a runnable AlphaShaper executable exists on this platform. """
        return cls._resolve_executable() is not None

    def calculate_as(self, frame_num, alpha=-1, volume=False):
        """
        Calculate the alpha shape for frame <frame_num>
        :param frame_num:
        :param alpha:
        :param volume:
        :return:
        """
        temp_file_name = f'./.temp_frame_{frame_num}.xyz'
        temp_output_facets_file_name = f'output_facets_{frame_num}.txt'
        temp_output_cells_file_name = f'output_cells_{frame_num}.txt'
        temp_output_volume_file_name = f'output_volumes_{frame_num}.txt'
            
        output_file_suffix = f'{frame_num}'
        alpha_shaper_exe = self._resolve_executable()
        if alpha_shaper_exe is None:
            raise RuntimeError(
                f'Alpha-shape calculation requires the compiled AlphaShaper executable, '
                f'which was not found for platform {platform.system()!r}. Build it from '
                f'PUCHIK/grid_project/alpha_shaper/src, set the ALPHA_SHAPER_EXECUTABLE '
                f'environment variable, or use the default convex-hull analysis '
                f'(use_alpha_shape=False).'
            )

        np.savetxt(temp_file_name, self.points, header=f'{len(self.points)}', comments='')
        proc = subprocess.run([alpha_shaper_exe, temp_file_name, f'{alpha}', output_file_suffix],
                              capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(f"AlphaShaper failed:\n{proc.stderr}")

        self.simplices = np.loadtxt(temp_output_facets_file_name, dtype=int)
        self.cells = np.loadtxt(temp_output_cells_file_name, dtype=int)
        self.volume = np.loadtxt(temp_output_volume_file_name, dtype=float) if volume else None

        os.remove(temp_file_name)
        os.remove(temp_output_facets_file_name)
        os.remove(temp_output_cells_file_name)
        os.remove(temp_output_volume_file_name)

        return self
