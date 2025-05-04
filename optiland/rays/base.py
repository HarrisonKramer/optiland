"""Base Rays

This module contains the base class for rays defined in a 3D space.

Kramer Harrison, 2024
"""

import optiland.backend as be


class BaseRays:
    """Base class for rays in a 3D space.

    Attributes:
        x (float): x-coordinate of the ray.
        y (float): y-coordinate of the ray.
        z (float): z-coordinate of the ray.

    """

    def translate(self, dx: float, dy: float, dz: float):
        """Shifts the rays in the x, y, and z directions.

        Args:
            dx (float): The amount to shift the rays in the x direction.
            dy (float): The amount to shift the rays in the y direction.
            dz (float): The amount to shift the rays in the z direction.

        """
        dx = be.array(dx)
        dy = be.array(dy)
        dz = be.array(dz)
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.z + dz
