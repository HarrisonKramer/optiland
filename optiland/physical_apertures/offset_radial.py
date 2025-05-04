"""Offset Radial Aperture Module

This module contains the OffsetRadialAperture class, which represents a radial
aperture that clips rays based on their distance from the origin, with an
offset in the x and y directions.

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.physical_apertures.radial import RadialAperture


class OffsetRadialAperture(RadialAperture):
    """Represents a radial aperture that clips rays based on their distance from
    the origin, with an offset in the x and y directions.

    Attributes:
        r_max (float): The maximum radius allowed for the rays.
        r_min (float): The minimum radius allowed for the rays. Defaults to 0.
        offset_x (float): The x-coordinate of the aperture's center.
        offset_y (float): The y-coordinate of the aperture's center.

    """

    def __init__(self, r_max, r_min=0, offset_x=0, offset_y=0):
        super().__init__(r_max, r_min)
        self.offset_x = offset_x
        self.offset_y = offset_y

    @property
    def extent(self):
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions.

        """
        return (
            self.offset_x - self.r_max,
            self.offset_x + self.r_max,
            self.offset_y - self.r_max,
            self.offset_y + self.r_max,
        )

    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (be.ndarray): The x-coordinate of the point.
            y (be.ndarray): The y-coordinate of the point.

        Returns:
            be.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        radius2 = (x - self.offset_x) ** 2 + (y - self.offset_y) ** 2
        return be.logical_and(radius2 <= self.r_max**2, radius2 >= self.r_min**2)

    def scale(self, scale_factor):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        super().scale(scale_factor)
        self.offset_x = self.offset_x * scale_factor
        self.offset_y = self.offset_y * scale_factor

    def to_dict(self):
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        aperture_dict = super().to_dict()
        aperture_dict["offset_x"] = self.offset_x
        aperture_dict["offset_y"] = self.offset_y
        return aperture_dict

    @classmethod
    def from_dict(cls, data):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            OffsetRadialAperture: The aperture object.

        """
        return cls(data["r_max"], data["r_min"], data["offset_x"], data["offset_y"])
