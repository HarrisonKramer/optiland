"""Rectangular Aperture Module

This module contains the RectangularAperture class, which represents a
rectangular aperture that clips rays based on their position.

Kramer Harrison, 2025
"""

from optiland.physical_apertures.base import BaseAperture


class RectangularAperture(BaseAperture):
    """Represents a rectangular aperture that clips rays based on their position.

    Attributes:
        x_min (float): The minimum x-coordinate allowed for the rays.
        x_max (float): The maximum x-coordinate allowed for the rays.
        y_min (float): The minimum y-coordinate allowed for the rays.
        y_max (float): The maximum y-coordinate allowed for the rays.

    """

    def __init__(self, x_min, x_max, y_min, y_max):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    @property
    def extent(self):
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions.

        """
        return self.x_min, self.x_max, self.y_min, self.y_max

    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        return (
            (self.x_min <= x)
            & (x <= self.x_max)
            & (self.y_min <= y)
            & (y <= self.y_max)
        )

    def scale(self, scale_factor):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        self.x_min = self.x_min * scale_factor
        self.x_max = self.x_max * scale_factor
        self.y_min = self.y_min * scale_factor
        self.y_max = self.y_max * scale_factor

    def to_dict(self):
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        aperture_dict = super().to_dict()
        aperture_dict["x_min"] = self.x_min
        aperture_dict["x_max"] = self.x_max
        aperture_dict["y_min"] = self.y_min
        aperture_dict["y_max"] = self.y_max
        return aperture_dict

    @classmethod
    def from_dict(cls, data):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            RectangularAperture: The aperture object.

        """
        return cls(data["x_min"], data["x_max"], data["y_min"], data["y_max"])
