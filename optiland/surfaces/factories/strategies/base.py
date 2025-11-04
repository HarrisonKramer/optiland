"""Base Strategy Module for Surface Creation

This module defines the abstract base class for surface creation strategies.
Each strategy encapsulates the logic for creating a specific type of surface
(e.g., standard, paraxial, object).

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.interactions.base import BaseInteractionModel
    from optiland.surfaces.factories.geometry_factory import GeometryFactory
    from optiland.surfaces.factories.interaction_model_factory import (
        InteractionModelFactory,
    )
    from optiland.surfaces.generics import BaseGeometry
    from optiland.surfaces.object_surface import ObjectSurface
    from optiland.surfaces.standard_surface import Surface


class BaseSurfaceStrategy(ABC):
    """Abstract base class for defining a surface creation strategy."""

    @abstractmethod
    def create_geometry(
        self, factory: GeometryFactory, cs: CoordinateSystem, config: dict
    ) -> BaseGeometry:
        """Creates the geometry for the surface.

        Args:
            factory: The geometry factory to use.
            cs: The coordinate system for the geometry.
            config: The configuration dictionary for the surface.

        Returns:
            The created geometry instance.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def create_interaction_model(
        self, factory: InteractionModelFactory, config: dict, **kwargs
    ) -> BaseInteractionModel | None:
        """Creates the interaction model for the surface.

        Args:
            factory: The interaction model factory to use.
            config: The configuration dictionary for the surface.
            **kwargs: Additional context required for creating the model.

        Returns:
            The created interaction model instance, or None.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_surface_class(self) -> type[Surface | ObjectSurface]:
        """Returns the appropriate surface class (e.g., Surface, ObjectSurface).

        Returns:
            The class type for the surface.
        """
        raise NotImplementedError  # pragma: no cover
