"""Decenter Variable Module

This module contains the DecenterVariable class, which represents a variable
for the decenter of an optic surface. The decenter variable can be used to
optimize the decenter of a surface in the optical system.

Kramer Harrison, 2024
"""

from optiland.optimization.variable.base import VariableBehavior


class DecenterVariable(VariableBehavior):
    """Represents a variable for the decenter of an optic surface.

    Args:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.
        axis (str): The axis of the decenter. Valid values are 'x' and 'y'.
        apply_scaling (bool): Whether to apply scaling to the variable.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.

    Methods:
        get_value(): Returns the current decenter value of the surface.
        update_value(new_value): Updates the decenter value of the surface.

    """

    def __init__(self, optic, surface_number, axis, apply_scaling=True, **kwargs):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)
        self.axis = axis

        if self.axis not in ["x", "y"]:
            raise ValueError(f'Invalid axis "{self.axis}" for decenter variable.')

    def get_value(self):
        """Returns the current decenter value of the surface.

        Returns:
            float: The current decenter value.

        """
        surf = self._surfaces.surfaces[self.surface_number]
        if self.axis == "x":
            value = surf.geometry.cs.x
        elif self.axis == "y":
            value = surf.geometry.cs.y

        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value):
        """Updates the decenter value of the surface.

        Args:
            new_value (float): The new decenter value.

        """
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)

        surf = self._surfaces.surfaces[self.surface_number]
        if self.axis == "x":
            surf.geometry.cs.x = new_value
        elif self.axis == "y":
            surf.geometry.cs.y = new_value

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
        return f"Decenter {self.axis.upper()}, Surface {self.surface_number}"
