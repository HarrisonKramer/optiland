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
        y (be.Tensor): The y-coordinates of the rays.
        u (be.Tensor): The slopes of the rays in the y-z plane.
        z (be.Tensor): The z-coordinates of the rays (optical axis).
        wavelength (be.Tensor): The wavelength of each ray.
        x (be.Tensor): The x-coordinates of the rays (initialized to zero for
            paraxial approximation in the y-z plane).
        i (be.Tensor): A tensor representing the intensity of the rays,
            initialized to ones.
        w (be.Tensor): Alias for wavelength.

    """

    def __init__(self, y, u, z, wavelength):
        """Initializes a ParaxialRays object.

        Args:
            y (float | list[float] | be.Tensor): The initial y-coordinates of the rays.
            u (float | list[float] | be.Tensor): The initial slopes of the rays
                in the y-z plane.
            z (float | list[float] | be.Tensor): The initial z-coordinates of the rays.
            wavelength (float | list[float] | be.Tensor): The wavelength of each ray.
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
        """Rotates the paraxial rays about the x-axis.

        This transforms the y, z coordinates and the slope u of the rays.
        It assumes a rotation in the y-z plane.

        Args:
            rx (float): The rotation angle in radians.
        """
        rx = be.array(rx)
        y_new = self.y * be.cos(rx) - self.z * be.sin(rx)
        z_new = self.y * be.sin(rx) + self.z * be.cos(rx)

        # The direction vector in the y-z plane is (self.u, 1) before rotation.
        # After rotation, the new components M_new and N_new are:
        M_new = self.u * be.cos(rx) - be.sin(rx) # Effectively rotating (u, 1)
        N_new = self.u * be.sin(rx) + be.cos(rx)

        self.y = y_new
        self.z = z_new
        self.u = M_new / N_new # New slope is M_new / N_new

    def rotate_y(self, ry: float):
        """Rotates the rays about the y-axis by a given angle.

        Args:
            ry (float): The rotation angle in radians.

        Raises:
            NotImplementedError: This rotation is not applicable to the 1D
                ParaxialRays model as it would introduce x-components,
                violating the paraxial assumption in the y-z plane.
        """
        raise NotImplementedError(
            "rotate_y is not applicable to the 1D ParaxialRays model as it would introduce x-components."
        )

    def rotate_z(self, rz: float):
        """Rotates the rays about the z-axis by a given angle.

        Args:
            rz (float): The rotation angle in radians.

        Raises:
            NotImplementedError: This rotation is not applicable to the 1D
                ParaxialRays model as it primarily affects x-components or
                polarization, neither of which are fully modeled here.
        """
        raise NotImplementedError(
            "rotate_z is not applicable to the 1D ParaxialRays model as it would introduce x-components."
        )
