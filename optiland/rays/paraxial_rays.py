"""Paraxial Rays

This module contains the ParaxialRays class, which represents paraxial rays in
an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays.base import BaseRays

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland._types import ScalarOrArray


class ParaxialRays(BaseRays):
    """Class representing paraxial rays in an optical system.

    Attributes:
        y: The y-coordinate of the rays.
        u: The slope of the rays.
        z: The z-coordinate of the rays.
        wavelength: The wavelength of the rays.

    Methods:
        propagate(t): Propagates the rays by a given distance.

    """

    def __init__(
        self,
        y: ArrayLike,
        u: ArrayLike,
        z: ArrayLike,
        wavelength: ArrayLike,
    ):
        self.y = be.as_array_1d(y)
        self.z = be.as_array_1d(z)
        self.u = be.as_array_1d(u)
        self.x = be.zeros_like(self.y)
        self.i = be.ones_like(self.y)
        self.w = be.as_array_1d(wavelength)

    def propagate(self, t: ScalarOrArray):
        """Propagates the rays by a given distance.

        Args:
            t: The distance to propagate the rays.

        """
        self.y = self.y + t * self.u
        self.z = self.z + t

    def rotate_x(self, rx: ScalarOrArray):
        """Rotate the rays about the x-axis."""
        # pragma: no cover

    def rotate_y(self, ry: ScalarOrArray):
        """Rotate the rays about the y-axis."""
        # pragma: no cover

    def rotate_z(self, rz: ScalarOrArray):
        """Rotate the rays about the z-axis."""
        # pragma: no cover
