"""
Base Source Class Module

This module defines the abstract base class for all source types in Optiland.
All concrete source implementations must inherit from BaseSource and implement
the generate_rays method.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.rays import RealRays


class BaseSource(ABC):
    """
    Abstract base class for all source types in Optiland.

    This class defines the interface that all source implementations must follow.
    All concrete source classes must implement the generate_rays method.
    """

    @abstractmethod
    def generate_rays(self, num_rays: int) -> RealRays:
        """
        Generate a specified number of rays from this source.

        Args:
            num_rays (int): The number of rays to generate.

        Returns:
            RealRays: An object containing the generated rays with their
                     positions, directions, intensities, and wavelengths.

        Raises:
            ValueError: If num_rays is not a positive integer or if ray
                       generation fails.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        Return a string representation of the source.

        Returns:
            str: A string describing the source and its parameters.
        """
        pass
