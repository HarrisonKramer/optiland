"""Thickness Variable Module

This module contains the ThicknessVariable class, which represents a variable
for the thickness of an optic surface. The class inherits from the
VariableBehavior class.

Kramer Harrison, 2024
"""

from optiland.optimization.variable.base import VariableBehavior


class ThicknessVariable(VariableBehavior):
    """Represents a variable for the thickness of an optic surface.

    Args:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.
        apply_scaling (bool): Whether to apply scaling to the variable.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.

    Methods:
        get_value(): Returns the current thickness value of the surface.
        update_value(new_value): Updates the thickness value of the surface.

    """

    def __init__(self, optic, surface_number, apply_scaling=True, **kwargs):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)

    def get_value(self):
        """Returns the current thickness value of the surface.

        Returns:
            float: The current thickness value.

        """
        value = self._surfaces.get_thickness(self.surface_number)[0]
        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value):
        """Updates the thickness value of the surface.

        Args:
            new_value (float): The new thickness value.

        """
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)
        self.optic.set_thickness(new_value, self.surface_number)

    def scale(self, value):
        """Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale

        """
        return value / 10.0 - 1.0

    def inverse_scale(self, scaled_value):
        """Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return (scaled_value + 1.0) * 10.0

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Thickness, Surface {self.surface_number}"
