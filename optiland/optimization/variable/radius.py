"""Radius of Curvature Variable Module

This module contains the RadiusVariable class, which represents a variable for
the radius of a surface in an optic. The class inherits from the
VariableBehavior class.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.variable.base import VariableBehavior


class RadiusVariable(VariableBehavior):
    """Represents a variable for the radius of a surface in an optic.

    Args:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            a linear scaler with factor=1/100 and offset=-1.0.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.

    Methods:
        get_value(): Returns the current value of the radius.
        update_value(new_value): Updates the value of the radius.

    """

    def __init__(self, optic, surface_number, scaler=None, **kwargs):
        if scaler is None:
            scaler = LinearScaler(factor=1 / 100.0, offset=-1.0)
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)

    def get_value(self):
        """Returns the current value of the radius.

        Returns:
            float: The current value of the radius.

        """
        return self._surfaces.radii[self.surface_number]

    def update_value(self, new_value):
        """Updates the value of the radius.

        Args:
            new_value (float): The new value of the radius.

        """
        self.optic.set_radius(new_value, self.surface_number)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Radius of Curvature, Surface {self.surface_number}"
