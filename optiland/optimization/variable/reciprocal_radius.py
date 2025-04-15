"""Reciprocal Radius Variable Module

This module contains the ReciprocalRadiusVariable class, which represents a variable for
the reciprocal of the radius of a surface in an optic. The transformation is useful
because a nearly flat surface (with a very large positive or negative radius) has a
reciprocal near zero, which makes the optimization around a nearly flat surface
continuous. The reciprocal radius provides a continuous transition when surface changes
from convex to concave (when the radius changes sign).


Daniel Miranda, 2025
ALL rights ceded to Kramer Harrison
"""

import numpy as np

from optiland.optimization.variable.base import VariableBehavior


class ReciprocalRadiusVariable(VariableBehavior):
    """Represents a variable for the reciprocal of the radius of a surface in an optic.

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
        get_value(): Returns the current value of the reciprocal radius.
        update_value(new_value): Updates the value via the reciprocal, converting back
         to radius.
    """

    def __init__(self, optic, surface_number, apply_scaling=True, **kwargs):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)

    def get_value(self):
        """Returns the current value of the reciprocal of the radius.

        Returns:
            float: The current reciprocal radius.
        """
        radius = self._surfaces.radii[self.surface_number]
        # Avoid division by zero
        if radius != 0:
            reciprocal = 1.0 / radius if np.isfinite(radius) else 0.0
        else:
            reciprocal = np.inf
        if self.apply_scaling:
            return self.scale(reciprocal)
        return reciprocal

    def update_value(self, new_value):
        """Updates the surface radius based on the new reciprocal variable value.

        Args:
            new_value (float): The new reciprocal radius value.
        """
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)
        # Allow zero but handle appropriately
        new_radius = np.inf if new_value == 0 else 1.0 / new_value
        self.optic.set_radius(new_radius, self.surface_number)

    def scale(self, value):
        """Scale the reciprocal value for improved optimization performance.

        Args:
            value (float): The reciprocal value to scale.

        Returns:
            float: The scaled reciprocal value.
        """
        return value * 10.0

    def inverse_scale(self, scaled_value):
        """Inverse scale the reciprocal value.

        Args:
            scaled_value (float): The scaled reciprocal value to inverse scale.

        Returns:
            float: The original reciprocal value.
        """
        return scaled_value / 10.0

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the reciprocal radius variable.
        """
        s = "scaled" if self.apply_scaling else "unscaled"
        n = self.surface_number
        v = self.get_value()

        return f"Reciprocal Radius of Curvature - Surface {n} - {s} value: {v}"
