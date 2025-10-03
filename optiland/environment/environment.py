"""Environment module

This module defines the Environment class, which encapsulates the
configuration for a specific optical environment. The Environment class
includes attributes for the immersion medium (a material instance) and the
physical conditions (temperature, pressure) of the environment.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
