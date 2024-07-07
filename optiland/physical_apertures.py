"""Optiland Physical Apertures Module

This module provides classes to define physical apertures on optical surfaces.

Kramer Harrison, 2024
"""


class BaseAperture:
    """
    Base class for physical apertures.

    Methods:
        clip(RealRays): Clips the given rays based on the aperture's shape.
    """

    def clip(self, rays):
        """
        Clips the given rays based on the aperture's shape.

        Parameters:
            rays (RealRays): List of rays to be clipped.

        Returns:
            list: List of clipped rays.
        """
        return rays


class RadialAperture(BaseAperture):
    """
    Represents a radial aperture that clips rays based on their distance from
    the origin.

    Attributes:
        r_max (float): The maximum radius allowed for the rays.
        r_min (float): The minimum radius allowed for the rays. Defaults to 0.
    """

    def __init__(self, r_max, r_min=0):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min

    def clip(self, rays):
        """
        Clips the given rays based on their distance from the origin.

        Args:
            rays (Rays): The rays to be clipped.
        """
        radius2 = rays.x**2 + rays.y**2
        condition = (radius2 > self.r_max**2) | (radius2 < self.r_min**2)
        rays.clip(condition)
