"""Polynomial XY Variable Module

This module contains the PolynomialCoeffVariable class, which represents a
variable for a polynomial coefficient of a PolynomialGeometry. The class
inherits from the VariableBehavior class.

Kramer Harrison, 2024
"""

import numpy as np

from optiland.optimization.variable.base import VariableBehavior


class PolynomialCoeffVariable(VariableBehavior):
    """Represents a variable for a polynomial coefficient of a PolynomialGeometry.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The index of the surface in the optical system.
        coeff_index (tuple(int, int)): The (x, y) indices of the polynomial
            coefficient.
        apply_scaling (bool): Whether to apply scaling to the variable.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Attributes:
        coeff_number (int): The index of the polynomial coefficient.

    """

    def __init__(
        self,
        optic,
        surface_number,
        coeff_index,
        apply_scaling=True,
        **kwargs,
    ):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)
        self.coeff_index = coeff_index

    def get_value(self):
        """Get the current value of the polynomial coefficient.

        Returns:
            float: The current value of the polynomial coefficient.

        """
        surf = self._surfaces.surfaces[self.surface_number]
        i, j = self.coeff_index
        try:
            value = surf.geometry.c[i][j]
        except IndexError:
            pad_width_i = max(0, i + 1 - surf.geometry.c.shape[0])
            pad_width_j = max(0, j + 1 - surf.geometry.c.shape[1])
            c_new = np.pad(
                surf.geometry.c,
                pad_width=((0, pad_width_i), (0, pad_width_j)),
                mode="constant",
                constant_values=0,
            )
            surf.geometry.c = c_new
            value = 0
        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value):
        """Update the value of the polynomial coefficient.

        Args:
            new_value (float): The new value of the polynomial coefficient.

        """
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)
        surf = self.optic.surface_group.surfaces[self.surface_number]
        i, j = self.coeff_index
        try:
            surf.geometry.c[i][j] = new_value
        except IndexError:
            pad_width_i = max(0, i + 1 - surf.geometry.c.shape[0])
            pad_width_j = max(0, j + 1 - surf.geometry.c.shape[1])
            c_new = np.pad(
                surf.geometry.c,
                pad_width=((0, pad_width_i), (0, pad_width_j)),
                mode="constant",
                constant_values=0,
            )
            c_new[i][j] = new_value
            surf.geometry.c = c_new

    def scale(self, value):
        """Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale

        """
        return value

    def inverse_scale(self, scaled_value):
        """Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return scaled_value

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Poly. Coeff. {self.coeff_index}, Surface {self.surface_number}"
