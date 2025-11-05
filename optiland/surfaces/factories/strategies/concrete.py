"""Concrete Strategies for Surface Creation

This module provides concrete implementations of the BaseSurfaceStrategy. Each
class encapsulates the specific logic required to create the components for a
particular type of optical surface.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optiland.surfaces.factories.strategies.base import BaseSurfaceStrategy
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.standard_surface import Surface

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.interactions.base import BaseInteractionModel
    from optiland.surfaces.factories.geometry_factory import GeometryFactory
    from optiland.surfaces.factories.interaction_model_factory import (
        InteractionModelFactory,
    )
    from optiland.surfaces.generics import BaseGeometry


class StandardStrategy(BaseSurfaceStrategy):
    """Strategy for creating standard refractive/reflective surfaces."""

    def create_geometry(
        self, factory: GeometryFactory, cs: CoordinateSystem, config: dict
    ) -> BaseGeometry:
        surface_type = config.get("surface_type", "standard")
        config_no_surface_type = dict(config)
        config_no_surface_type.pop("surface_type", None)
        return factory.create(surface_type, cs, **config_no_surface_type)

    def create_interaction_model(
        self, factory: InteractionModelFactory, config: dict, **kwargs: Any
    ) -> BaseInteractionModel | None:
        if config.get("phase_profile") is not None:
            interaction_type = "phase"
            kwargs["phase_profile"] = config.get("phase_profile")
        else:
            interaction_type = "refractive_reflective"
        return factory.create(interaction_type=interaction_type, **kwargs)

    def get_surface_class(self) -> type[Surface]:
        return Surface


class ParaxialStrategy(BaseSurfaceStrategy):
    """Strategy for creating paraxial surfaces (thin lenses)."""

    def create_geometry(
        self, factory: GeometryFactory, cs: CoordinateSystem, config: dict
    ) -> BaseGeometry:
        config.pop("surface_type", None)
        return factory.create("paraxial", cs, **config)

    def create_interaction_model(
        self, factory: InteractionModelFactory, config: dict, **kwargs: Any
    ) -> BaseInteractionModel | None:
        kwargs["focal_length"] = config.get("f")
        return factory.create(interaction_type="thin_lens", **kwargs)

    def get_surface_class(self) -> type[Surface]:
        return Surface


class GratingStrategy(BaseSurfaceStrategy):
    """Strategy for creating diffractive grating surfaces."""

    def create_geometry(
        self, factory: GeometryFactory, cs: CoordinateSystem, config: dict
    ) -> BaseGeometry:
        config.pop("surface_type", None)
        return factory.create("grating", cs, **config)

    def create_interaction_model(
        self, factory: InteractionModelFactory, config: dict, **kwargs: Any
    ) -> BaseInteractionModel | None:
        return factory.create(interaction_type="diffractive", **kwargs)

    def get_surface_class(self) -> type[Surface]:
        return Surface


class ObjectStrategy(BaseSurfaceStrategy):
    """Strategy for creating the object surface."""

    def create_geometry(
        self, factory: GeometryFactory, cs: CoordinateSystem, config: dict
    ) -> BaseGeometry:
        # The Object Surface can have any geometry. Default to 'plane' if not specified.
        surface_type = config.pop("surface_type", "plane")
        return factory.create(surface_type, cs, **config)

    def create_interaction_model(
        self, factory: InteractionModelFactory, config: dict, **kwargs: Any
    ) -> None:
        return None

    def get_surface_class(self) -> type[ObjectSurface]:
        return ObjectSurface
