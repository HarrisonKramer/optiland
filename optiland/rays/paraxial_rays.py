"""Paraxial Rays

This module contains the ParaxialRays class, which represents paraxial rays in
an optical system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.rays.base import BaseRays


class ParaxialRays(BaseRays):
    """Represents a collection of paraxial rays.

    Paraxial rays are rays that are close to the optical axis and make small
    angles with it. This class provides methods for propagating and manipulating
    these rays.

    Attributes:
        y (be.ndarray): The y-coordinates of the rays.
        u (be.ndarray): The slopes of the rays in the y-z plane.
        z (be.ndarray): The z-coordinates of the rays (optical axis).
        wavelength (be.ndarray): The wavelength of each ray.
        x (be.ndarray): The x-coordinates of the rays (initialized to zero for
            paraxial approximation in the y-z plane).
        i (be.ndarray): A tensor representing the intensity of the rays,
            initialized to ones.
        w (be.ndarray): Alias for wavelength.

    """

    def __init__(self, y, u, z, wavelength):
        """Initializes a ParaxialRays object.

        Args:
            y (float | list[float] | be.ndarray): The initial y-coordinates of the rays.
            u (float | list[float] | be.ndarray): The initial slopes of the rays
                in the y-z plane.
            z (float | list[float] | be.ndarray): The initial z-coordinates of the rays.
            wavelength (float | list[float] | be.ndarray): The wavelength of each ray.
        """
        self.y = be.as_array_1d(y)
        self.z = be.as_array_1d(z)
        self.u = be.as_array_1d(u)
        self.x = be.zeros_like(self.y)
        self.i = be.ones_like(self.y)
        self.w = be.as_array_1d(wavelength)

    def propagate(self, t: float):
        """Propagates the rays along the optical (z) axis by a distance `t`.

        Updates the y-coordinate based on the slope `u` and the propagation
        distance `t`. The z-coordinate is incremented by `t`.

        Args:
            t (float): The distance to propagate the rays.
        """
        self.y = self.y + t * self.u
        self.z = self.z + t

    def rotate_x(self, rx: float):
        """Rotate the rays about the x-axis."""
        # pragma: no cover

    def rotate_y(self, ry: float):
        """Rotate the rays about the y-axis."""
        # pragma: no cover

    def rotate_z(self, rz: float):
        """Rotate the rays about the z-axis."""
        # pragma: no cover
