"""Base Rays

This module contains the base class for rays defined in a 3D space.

Kramer Harrison, 2024
"""

import optiland.backend as be


class BaseRays:
    """Base class for representing a collection of rays in a 3D space.

    This class serves as a foundation for more specialized ray types, providing
    basic attributes and methods for manipulating ray positions.

    Attributes:
        x (be.Tensor): A tensor representing the x-coordinates of the rays.
        y (be.Tensor): A tensor representing the y-coordinates of the rays.
        z (be.Tensor): A tensor representing the z-coordinates of the rays.
            Typically, this represents the optical axis in an optical system.
    """

    def translate(self, dx: float, dy: float, dz: float):
        """Translates the rays by a given displacement in x, y, and z.

        Args:
            dx (float): The displacement in the x-direction.
            dy (float): The displacement in the y-direction.
            dz (float): The displacement in the z-direction.
        """
        dx = be.array(dx)
        dy = be.array(dy)
        dz = be.array(dz)
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.z + dz
