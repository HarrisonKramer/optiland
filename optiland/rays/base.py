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
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.z + dz

    def _process_input(self, data):
        """Process the input data and convert it into a 1-dimensional NumPy array
        of floats.

        Args:
            data (int, float, or be.ndarray): The input data to be processed.

        Returns:
            be.ndarray: The processed data as a 1-dimensional NumPy array of
                floats.

        Raises:
            ValueError: If the input data type is not supported (must be a
                scalar or a NumPy array).

        """
        if isinstance(data, (int, float)):
            return be.array([data])
        elif isinstance(data, list):
            return be.atleast_1d(data)
        if be.is_array_like(data):
            return be.ravel(data)
        raise ValueError(
            "Unsupported input type. Must be a scalar, a list, or array-like.",
        )
