"""Base Variable Module

This module contains the base class for a variable in an optic system. The
VariableBehavior class is an abstract class that represents the behavior of a
variable in an optic system. It is used as a base class for all variables in
the optimization process.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod


class VariableBehavior(ABC):
    """Represents the behavior of a variable in an optic system.

    Args:
        optic (Optic): The optic system to which the variable belongs.
        surface_number (int): The surface number of the variable.
        apply_scaling (bool): Whether to apply scaling to the variable.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic system to which the variable belongs.
        _surfaces (SurfaceGroup): The group of surfaces in the optic system.
        surface_number (int): The surface number of the variable.

    """

    def __init__(self, optic, surface_number, apply_scaling=True, **kwargs):
        self.optic = optic
        self._surfaces = self.optic.surface_group
        self.surface_number = surface_number
        self.apply_scaling = apply_scaling

    @abstractmethod
    def get_value(self):
        """Get the value of the variable.

        Returns:
            The value of the variable.

        """
        # pragma: no cover

    @abstractmethod
    def update_value(self, new_value):
        """Update the value of the variable.

        Args:
            new_value: The new value of the variable.

        """
        # pragma: no cover

    def scale(self, value):
        """Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale

        """
        return value  # pragma: no cover

    def inverse_scale(self, scaled_value):
        """Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return scaled_value  # pragma: no cover
