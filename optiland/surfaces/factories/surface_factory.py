"""Surface Factory

This module contains the refactored SurfaceFactory, a stateless orchestrator for
creating surface objects. It uses Dependency Injection and the Strategy Pattern
to delegate component creation to specialized, SOLID-compliant factories and
strategies.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.surfaces.factories.coating_factory import CoatingFactory
    from optiland.surfaces.factories.coordinate_system_factory import (
        CoordinateSystemFactory,
    )
    from optiland.surfaces.factories.geometry_factory import GeometryFactory
    from optiland.surfaces.factories.interaction_model_factory import (
        InteractionModelFactory,
    )
    from optiland.surfaces.factories.material_factory import MaterialFactory
    from optiland.surfaces.factories.strategy_provider import SurfaceStrategyProvider
    from optiland.surfaces.factories.types import SurfaceContext
    from optiland.surfaces.object_surface import ObjectSurface
    from optiland.surfaces.standard_surface import Surface


class SurfaceFactory:
    """A stateless orchestrator for creating surfaces using injected dependencies.

    This factory relies on the Strategy Pattern to handle variations in surface
    creation logic. It is initialized with all necessary component factories
    and a strategy provider, making it decoupled and easily testable.
    """

    def __init__(
        self,
        cs_factory: CoordinateSystemFactory,
        geom_factory: GeometryFactory,
        mat_factory: MaterialFactory,
        coat_factory: CoatingFactory,
        int_factory: InteractionModelFactory,
        strategy_provider: SurfaceStrategyProvider,
    ):
        self._cs_factory = cs_factory
        self._geom_factory = geom_factory
        self._mat_factory = mat_factory
        self._coat_factory = coat_factory
        self._int_factory = int_factory
        self._strategy_provider = strategy_provider

    def create_surface(
        self, config: dict[str, Any], context: SurfaceContext
    ) -> Surface | ObjectSurface:
        """Creates a surface object based on a configuration and context.

        Args:
            config: A dictionary containing the surface's specification
                    (e.g., surface_type, material, radius).
            context: A data object containing the stateful context for the
                     creation (e.g., index, z-position, precedent material).

        Returns:
            The created surface or object surface instance.
        """
        surface_type = config.get("surface_type")
        strategy = self._strategy_provider.get_strategy(surface_type)

        cs = self._cs_factory.create(
            x=config.get("dx", 0.0),
            y=config.get("dy", 0.0),
            z=context.z,
            rx=config.get("rx", 0.0),
            ry=config.get("ry", 0.0),
            rz=config.get("rz", 0.0),
        )

        material_spec = config.get("material")
        material_post = self._mat_factory.create(material_spec)

        is_reflective = material_spec == "mirror"
        if is_reflective:
            material_post = context.material_pre

        coating = self._coat_factory.create(
            config.get("coating"), context.material_pre, material_post
        )

        geometry = strategy.create_geometry(self._geom_factory, cs, config)

        # Create model with parent_surface=None initially.
        # It will be linked after the surface object is created.
        interaction_model = strategy.create_interaction_model(
            self._int_factory,
            config,
            parent_surface=None,
            is_reflective=is_reflective,
            coating=coating,
            bsdf=config.get("bsdf"),
        )

        surface_class = strategy.get_surface_class()

        surface_obj = surface_class(
            geometry=geometry,
            material_post=material_post,
            comment=config.get("comment", ""),
            **(
                {
                    "is_stop": config.get("is_stop", False),
                    "surface_type": surface_type or "standard",
                    "aperture": config.get("aperture"),
                    "interaction_model": interaction_model,
                    "previous_surface": None,
                }
                if surface_class.__name__ == "Surface"
                else {}
            ),
        )

        # Link the interaction model back to its parent surface
        if interaction_model:
            interaction_model.parent_surface = surface_obj

        surface_obj.thickness = config.get("thickness", 0.0)
        return surface_obj
