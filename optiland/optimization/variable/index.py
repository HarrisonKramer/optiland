"""Refractive Index Variable Module

This module contains the IndexVariable class, which represents a variable for
the index of refraction at a specific surface and wavelength. The variable can
be used in optimization problems to optimize the index of refraction at a
specific surface.

Kramer Harrison, 2024
"""

from optiland.optimization.variable.base import VariableBehavior


class IndexVariable(VariableBehavior):
    """Represents a variable for the index of refraction at a specific surface
    and wavelength.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The surface number where the variable is applied.
        wavelength (float): The wavelength at which the index of refraction is
            calculated.
        apply_scaling (bool): Whether to apply scaling to the variable.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.
        wavelength (float): The wavelength at which the index of refraction is
            calculated.

    Methods:
        get_value(): Returns the value of the index of refraction at the
            specified surface and wavelength.
        update_value(new_value): Updates the value of the index of refraction
            at the specified surface.

    """

    def __init__(self, optic, surface_number, wavelength, apply_scaling=True, **kwargs):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)
        self.wavelength = wavelength

    def get_value(self):
        """Returns the value of the index of refraction at the specified surface
        and wavelength.

        Returns:
            float: The value of the index of refraction.

        """
        n = self.optic.n(self.wavelength)
        value = n[self.surface_number]
        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value):
        """Updates the value of the index of refraction at the specified surface.

        Args:
            new_value (float): The new value of the index of refraction.

        """
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)
        self.optic.set_index(new_value, self.surface_number)

    def scale(self, value):
        """Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale

        """
        return value - 1.5

    def inverse_scale(self, scaled_value):
        """Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return scaled_value + 1.5

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Refractive Index, Surface {self.surface_number}"
