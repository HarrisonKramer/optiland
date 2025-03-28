"""Polygon Aperture Module

This module contains the PolygonAperture class, which represents a polygonal
aperture that clips rays based on a set of vertices. It also contains the
FileAperture class, which reads an aperture definition from a file and creates
a polygon-based aperture.

Kramer Harrison, 2025
"""

import numpy as np
from matplotlib.path import Path

from optiland.physical_apertures.base import BaseAperture


class PolygonAperture(BaseAperture):
    """Represents a polygonal aperture that clips rays based on their position.

    Attributes:
        x (list or np.ndarray): x-coordinates of the polygon's vertices.
        y (list or np.ndarray): y-coordinates of the polygon's vertices.
        vertices (np.ndarray): Array-like of shape (n, 2) defining the
            polygon vertices.

    """

    def __init__(self, x, y):
        super().__init__()
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.vertices = np.column_stack((self.x, self.y))
        self._path = Path(self.vertices)

    @property
    def extent(self):
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions.

        """
        return (self.x.min(), self.x.max(), self.y.min(), self.y.max())

    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        x = np.array(x)
        y = np.array(y)
        points = np.column_stack((x.ravel(), y.ravel()))
        return self._path.contains_points(points).reshape(x.shape)

    def scale(self, scale_factor):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        self.vertices *= scale_factor
        self._path = Path(self.vertices)

    def to_dict(self):
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        aperture_dict = super().to_dict()
        aperture_dict["x"] = self.x
        aperture_dict["y"] = self.y
        return aperture_dict

    @classmethod
    def from_dict(cls, data):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            PolygonAperture: The aperture object.

        """
        return cls(data["x"], data["y"])


class FileAperture(PolygonAperture):
    """Reads an aperture definition from a file and creates a polygon-based
    aperture.

    The file should contain two columns representing the x and y coordinates,
    respectively. It supports various file formats (e.g. CSV, TXT) by allowing
    you to specify a delimiter and the number of header lines to skip. Comments
    can be added to the file by starting a line with '//'.

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
        self.filepath = filepath
        self.delimiter = delimiter
        self.skip_header = skip_header
        x, y = self._load_vertices(filepath, delimiter, skip_header)
        super().__init__(x, y)

    def _load_vertices(self, filepath, delimiter, skip_header):
        """Load x and y vertices from the specified file.

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
        encodings = [
            "utf-8",
            "utf-16",
            "utf-16le",
            "utf-16be",
            "utf-32",
            "utf-32le",
            "utf-32be",
            "latin1",
            "ascii",
        ]
        data = None
        for encoding in encodings:
            try:
                with open(filepath, encoding=encoding) as f:
                    # delimiter defaults to space if not specified
                    delim = delimiter if delimiter is not None else " "
                    data = np.genfromtxt(
                        f,
                        delimiter=delim,
                        comments="//",
                        skip_header=skip_header,
                    )
                if data is not None:
                    break
            except UnicodeDecodeError:
                continue

        if data is None or data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f'Error reading aperture file "{filepath}"')

        x = data[:, 0]
        y = data[:, 1]
        return x, y

    def to_dict(self):
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        aperture_dict = super().to_dict()
        aperture_dict["filepath"] = self.filepath
        aperture_dict["delimiter"] = self.delimiter
        aperture_dict["skip_header"] = self.skip_header
        return aperture_dict

    @classmethod
    def from_dict(cls, data):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            FileAperture: The aperture object.

        """
        return cls(data["filepath"], data["delimiter"], data["skip_header"])
