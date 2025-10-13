"""Environment module

This module defines the Environment class, which encapsulates the
configuration for a specific optical environment. The Environment class
includes attributes for the immersion medium (a material instance) and the
physical conditions (temperature, pressure) of the environment.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial

    from .conditions import EnvironmentalConditions


class Environment:
    """
    A data class that holds the configuration for a specific
    optical environment.

    Attributes:
        medium (BaseMaterial): The material instance representing the
            immersion medium (e.g., Air, Water, Vacuum).
        conditions (EnvironmentalConditions): The physical conditions
            (temperature, pressure) of the environment.
    """

    def __init__(self, medium: BaseMaterial, conditions: EnvironmentalConditions):
        self.medium = medium
        self.conditions = conditions

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the environment.

        Returns:
            dict: A dictionary containing the medium and conditions.
        """
        return {
            "medium": self.medium.to_dict(),
            "conditions": self.conditions.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Environment:
        """Creates an Environment instance from a dictionary.

        Args:
            data (dict): A dictionary containing the medium and conditions.

        Returns:
            Environment: A new Environment instance.
        """
        from optiland.materials.base import BaseMaterial

        from .conditions import EnvironmentalConditions

        medium = BaseMaterial.from_dict(data["medium"])
        conditions = EnvironmentalConditions.from_dict(data["conditions"])
        return cls(medium, conditions)
