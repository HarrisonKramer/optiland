"""Surface Strategy Provider

This module contains the SurfaceStrategyProvider, which is responsible for
selecting the appropriate surface creation strategy based on the surface type.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.surfaces.factories.strategies.concrete import (
    GratingStrategy,
    ObjectStrategy,
    ParaxialStrategy,
    StandardStrategy,
)

if TYPE_CHECKING:
    from optiland.surfaces.factories.strategies.base import BaseSurfaceStrategy


class SurfaceStrategyProvider:
    """Provides the correct surface creation strategy based on surface type."""

    def __init__(self):
        self._strategies: dict[str, BaseSurfaceStrategy] = {
            "standard": StandardStrategy(),
            "paraxial": ParaxialStrategy(),
            "object": ObjectStrategy(),
            "grating": GratingStrategy(),
        }
        self._default_strategy = self._strategies["standard"]

    def get_strategy(self, surface_type: str | None) -> BaseSurfaceStrategy:
        """Returns the strategy for the given surface type.

        Args:
            surface_type: The type of the surface (e.g., 'paraxial', 'object').

        Returns:
            The corresponding strategy instance. Falls back to StandardStrategy
            for geometric types that don't have a special strategy.
        """
        if surface_type is None:
            return self._default_strategy
        return self._strategies.get(surface_type, self._default_strategy)
