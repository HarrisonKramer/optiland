"""Surface Strategy Provider

This module contains the SurfaceStrategyProvider, which is responsible for
selecting the appropriate surface creation strategy based on the surface type.

Kramer Harrison, 2025
"""

from __future__ import annotations

from optiland.surfaces.factories.strategies.base import BaseSurfaceStrategy
from optiland.surfaces.factories.strategies.concrete import (
    GratingStrategy,
    ObjectStrategy,
    ParaxialStrategy,
    StandardStrategy,
)


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

    def get_strategy(
        self, surface_type: str | None, index: int
    ) -> BaseSurfaceStrategy:
        """Returns the strategy for the given surface type and index.

        The `index` is used to enforce the ObjectStrategy for the first surface.

        Args:
            surface_type: The type of the surface (e.g., 'paraxial').
            index: The index of the surface being created.

        Returns:
            The corresponding strategy instance.
        """
        if index == 0:
            return self._strategies["object"]

        if surface_type is None:
            return self._default_strategy
        return self._strategies.get(surface_type, self._default_strategy)
