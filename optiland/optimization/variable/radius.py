"""Radius of Curvature Variable Module

This module contains the RadiusVariable class, which represents a variable for
the radius of a surface in an optic. The class inherits from the
VariableBehavior class.

Kramer Harrison, 2024
"""

from optiland.optimization.variable.base import VariableBehavior


class RadiusVariable(VariableBehavior):
    """Represents a variable for the radius of a surface in an optic.

    Args:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.
        apply_scaling (bool): Whether to apply scaling to the variable.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.

    Methods:
        get_value(): Returns the current value of the radius.
        update_value(new_value): Updates the value of the radius.

    """

    def __init__(self, optic, surface_number, apply_scaling=True, **kwargs):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)

    def get_value(self):
        """Returns the current value of the radius.

        Returns:
            float: The current value of the radius.

        """
        value = self._surfaces.radii[self.surface_number]
        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value):
        """Updates the value of the radius.

        Args:
            new_value (float): The new value of the radius.

        """
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)
        self.optic.set_radius(new_value, self.surface_number)

    def scale(self, value):
        """Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale

        """
        return value / 100.0 - 1.0

    def inverse_scale(self, scaled_value):
        """Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return (scaled_value + 1.0) * 100.0

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Radius of Curvature, Surface {self.surface_number}"
