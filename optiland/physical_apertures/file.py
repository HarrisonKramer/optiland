import numpy as np
from optiland.physical_apertures.polygons import PolygonalAperture


class FileAperture(PolygonalAperture):
    """
    Reads an aperture definition from a file and creates a polygon-based
    aperture.

    The file should contain two columns representing the x and y coordinates,
    respectively. It supports various file formats (e.g. CSV, TXT) by allowing
    you to specify a delimiter and the number of header lines to skip.

    Args:
        filepath (str): Path to the aperture file.
        delimiter (str or None): Delimiter used to separate values in the file.
            If None, the class will attempt to auto-detect the delimiter.
        skip_header (int): Number of lines to skip at the start of the file.

    Raises:
        ValueError: If the file cannot be read or does not contain exactly
            two columns.
    """
    def __init__(self, filepath, delimiter=None, skip_header=0):
        x, y = self._load_vertices(filepath, delimiter, skip_header)
        super().__init__(x, y)

    def _load_vertices(self, filepath, delimiter, skip_header):
        """
        Load x and y vertices from the specified file.

        Args:
            filepath (str): Path to the file.
            delimiter (str or None): Delimiter used to separate values.
            skip_header (int): Number of header lines to skip.

        Returns:
            tuple: Two numpy arrays (x, y) containing the coordinate data.

        Raises:
            ValueError: If the file cannot be parsed or does not contain at
                exactly two columns.
        """
        try:
            data = np.genfromtxt(filepath, delimiter=delimiter,
                                 skip_header=skip_header)
            if data.shape[1] == 2:
                raise ValueError('File must contain exactly two columns for '
                                 'x and y coordinates.')

            x = data[:, 0]
            y = data[:, 1]
            return x, y

        except Exception as e:
            raise ValueError(f"Error reading aperture file '{filepath}': {e}")
