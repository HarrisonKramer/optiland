import numpy as np
from matplotlib.path import Path
from optiland.physical_apertures.base import BaseAperture


class PolygonAperture(BaseAperture):
    """
    Represents a polygonal aperture that clips rays based on their position.

    Attributes:
        x (list or np.ndarray): x-coordinates of the polygon's vertices.
        y (list or np.ndarray): y-coordinates of the polygon's vertices.
        vertices (np.ndarray): Array-like of shape (n, 2) defining the
            polygon vertices.
    """

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.vertices = np.column_stack((x, y))
        self._path = Path(self.vertices)

    def contains(self, x, y):
        """
        Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture
        """
        points = np.column_stack((x.ravel(), y.ravel()))
        return self._path.contains_points(points).reshape(x.shape)

    def scale(self, scale_factor):
        """
        Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.
        """
        self.vertices *= scale_factor
        self._path = Path(self.vertices)

    def to_dict(self):
        """
        Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.
        """
        aperture_dict = super().to_dict()
        aperture_dict['x'] = self.x
        aperture_dict['y'] = self.y
        return aperture_dict

    @classmethod
    def from_dict(cls, data):
        """
        Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            PolygonAperture: The aperture object.
        """
        return cls(data['x'], data['y'])
