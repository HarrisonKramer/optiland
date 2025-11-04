"""Surface Group

This module contains the SurfaceGroup class, which represents a group of
surfaces in an optical system. The SurfaceGroup class provides methods for
tracing rays through the surfaces, adding and removing surfaces, and
converting the group to and from a dictionary for serialization.

Kramer Harrison, 2024
"""

from __future__ import annotations

from contextlib import suppress
from functools import cached_property
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.coatings import BaseCoatingPolarized
from optiland.materials.base import BaseMaterial
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
from optiland.surfaces.factories.surface_factory import SurfaceFactory
from optiland.surfaces.factories.types import SurfaceContext
from optiland.surfaces.standard_surface import Surface

if TYPE_CHECKING:
    from optiland._types import SurfaceType


class SurfaceGroup:
    """Represents a group of surfaces in an optical system."""

    def __init__(self, surfaces: list[Surface] | None = None):
        """Initializes a new instance of the SurfaceGroup class."""
        if surfaces is None:
            self._surfaces = []
        else:
            self._surfaces = surfaces
            self._update_surface_links()

        cs_factory = CoordinateSystemFactory()
        geom_factory = GeometryFactory()
        mat_factory = MaterialFactory()
        coat_factory = CoatingFactory()
        int_factory = InteractionModelFactory()
        strategy_provider = SurfaceStrategyProvider()

        self.surface_factory = SurfaceFactory(
            cs_factory,
            geom_factory,
            mat_factory,
            coat_factory,
            int_factory,
            strategy_provider,
        )

    def _update_surface_links(self):
        with suppress(KeyError):
            self.__dict__.pop("surfaces")
        surfaces = self._surfaces
        if surfaces:
            surfaces[0].previous_surface = None
            for idx, surface in enumerate(surfaces[1:]):
                surface.previous_surface = surfaces[idx]

    def __add__(self, other: SurfaceGroup) -> SurfaceGroup:
        offset = self.surfaces[-1].geometry.cs.z if self.surfaces else 0.0
        object_distance = other.surfaces[0].geometry.cs.z
        if be.isfinite(object_distance):
            offset = offset - object_distance
        for surf in other.surfaces[1:]:
            surf.geometry.cs.z = surf.geometry.cs.z + offset
        for surface in other.surfaces:
            surface.is_stop = False
        return SurfaceGroup(self._surfaces[:-1] + other._surfaces[1:])

    @cached_property
    def surfaces(self) -> tuple[Surface, ...]:
        return tuple(self._surfaces)

    def clear(self):
        self._surfaces = []
        self._update_surface_links()

    @property
    def x(self):
        return be.stack([surf.x for surf in self.surfaces if be.size(surf.x) > 0])

    @property
    def y(self):
        return be.stack([surf.y for surf in self.surfaces if be.size(surf.y) > 0])

    @property
    def z(self):
        return be.stack([surf.z for surf in self.surfaces if be.size(surf.z) > 0])

    @property
    def L(self):
        return be.stack([surf.L for surf in self.surfaces if be.size(surf.L) > 0])

    @property
    def M(self):
        return be.stack([surf.M for surf in self.surfaces if be.size(surf.M) > 0])

    @property
    def N(self):
        return be.stack([surf.N for surf in self.surfaces if be.size(surf.N) > 0])

    @property
    def opd(self):
        return be.stack([surf.opd for surf in self.surfaces if be.size(surf.opd) > 0])

    @property
    def u(self):
        return be.stack([surf.u for surf in self.surfaces if be.size(surf.u) > 0])

    @property
    def intensity(self):
        return be.stack(
            [surf.intensity for surf in self.surfaces if be.size(surf.intensity) > 0]
        )

    @property
    def positions(self):
        positions = be.array(
            [surf.geometry.cs.position_in_gcs[2] for surf in self.surfaces]
        )
        return positions.reshape(-1, 1)

    @property
    def stop_index(self) -> int:
        for index, surface in enumerate(self.surfaces):
            if surface.is_stop:
                return index
        raise ValueError("No stop surface found.")

    @stop_index.setter
    def stop_index(self, index: int):
        if not 1 <= index <= len(self.surfaces) - 2:
            raise ValueError("Index out of range")
        for idx, surf in enumerate(self.surfaces):
            surf.is_stop = index == idx

    @property
    def num_surfaces(self) -> int:
        return len(self.surfaces)

    def add_surface(self, index: int, **kwargs: Any):
        new_surface = kwargs.pop("new_surface", None)
        if new_surface:
            self._surfaces.insert(index, new_surface)
            self._rebuild_state_from_surfaces()
            return

        # State logic: Enforce 'object' surface type for index 0
        if index == 0:
            kwargs["surface_type"] = "object"
            material_pre = None
        else:
            material_pre = self.surfaces[index - 1].material_post

        thickness = kwargs.get("thickness", 0.0)
        if index == 0:
            z = -thickness
        elif index == 1:
            z = 0.0
        else:
            prev_surface = self.surfaces[index - 1]
            z = prev_surface.geometry.cs.z + prev_surface.thickness

        context = SurfaceContext(index=index, z=z, material_pre=material_pre)
        new_surface = self.surface_factory.create_surface(config=kwargs, context=context)

        self._surfaces.insert(index, new_surface)
        self._update_surface_links()

        if index < len(self._surfaces) - 1:
            self._update_coordinate_systems(start_index=index + 1)

    def _rebuild_state_from_surfaces(self):
        self._update_surface_links()

    def remove_surface(self, index: int):
        if index == 0:
            raise ValueError("Cannot remove object surface (index 0).")
        del self._surfaces[index]
        self._rebuild_state_from_surfaces()
        if index < len(self._surfaces):
             self._update_coordinate_systems(start_index=index)

    def reset(self):
        for surface in self.surfaces:
            surface.reset()

    def to_dict(self) -> dict:
        return {"surfaces": [surface.to_dict() for surface in self.surfaces]}

    @classmethod
    def from_dict(cls, data: dict) -> SurfaceGroup:
        sg = cls()
        surfaces = [Surface.from_dict(s_data) for s_data in data["surfaces"]]
        sg._surfaces = surfaces
        sg._rebuild_state_from_surfaces()
        return sg

    def _update_coordinate_systems(self, start_index: int):
        for i in range(start_index, len(self._surfaces)):
            prev_surface = self._surfaces[i - 1]
            new_z = prev_surface.geometry.cs.z + prev_surface.thickness
            self._surfaces[i].geometry.cs.z = be.array(new_z)

    def trace(self, rays, skip=0):
        self.reset()
        for surface in self.surfaces[skip:]:
            surface.trace(rays)
        return rays
