"""Conic Constant Variable Module

This module contains the class for a conic constant variable in an optic
system. The ConicVariable class is a subclass of the VariableBehavior class
that represents a variable for the conic constant of a surface in an optic
system. It is used in the optimization process for conic surfaces.

Kramer Harrison, 2024
"""

from optiland.optimization.variable.base import VariableBehavior


class ConicVariable(VariableBehavior):
    """Represents a variable for the conic constant of a surface in an optic.

    Args:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The index of the surface in the optic.
        apply_scaling (bool): Whether to apply scaling to the variable.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The index of the surface in the optic.

    Methods:
        get_value: Returns the current conic constant of the surface.
        update_value: Updates the conic value of the surface.

    """

    def __init__(self, optic, surface_number, apply_scaling=True, **kwargs):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)

    def get_value(self):
        """Returns the current conic constant of the surface.

        Returns:
            float: The conic constant of the surface.

        """
        return self._surfaces.conic[self.surface_number]

    def update_value(self, new_value):
        """Updates the conic value of the surface.

        Args:
            new_value (float): The new conic constant to set.

        """
        self.optic.set_conic(new_value, self.surface_number)

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
        return f"Conic Constant, Surface {self.surface_number}"
