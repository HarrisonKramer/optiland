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
from typing import Any

import optiland.backend as be
from optiland.coatings import BaseCoatingPolarized
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


class SurfaceGroup:
    """Represents a group of surfaces in an optical system."""

    def __init__(self, surfaces: list[Surface] | None = None):
        """Initializes a new instance of the SurfaceGroup class."""
        self._surfaces: list[Surface] = surfaces if surfaces is not None else []
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
        self._has_absolute_z = False

    def _update_surface_links(self):
        with suppress(KeyError):
            self.__dict__.pop("surfaces")
        if self._surfaces:
            self._surfaces[0].previous_surface = None
            for idx, surface in enumerate(self._surfaces[1:]):
                surface.previous_surface = self._surfaces[idx]

    @cached_property
    def surfaces(self) -> tuple[Surface, ...]:
        return tuple(self._surfaces)

    def clear(self):
        self._surfaces = []
        self._has_absolute_z = False
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
    def num_surfaces(self) -> int:
        return len(self.surfaces)

    @property
    def total_track(self):
        if self.num_surfaces < 2:
            raise ValueError("Not enough surfaces to calculate total track.")
        z = self.positions[1:]
        return be.max(z) - be.min(z)

    def add_surface(self, index: int, **kwargs: Any):
        if "z" in kwargs:
            if "thickness" in kwargs:
                raise ValueError('Cannot define both "thickness" and "z".')
            if "dx" in kwargs or "dy" in kwargs:
                raise ValueError(
                    'Cannot define "dx" or "dy" when using absolute "x", "y", "z".'
                )
            self._has_absolute_z = True
        elif self._has_absolute_z and "thickness" in kwargs:
            raise ValueError('Cannot pass "thickness" after defining "x", "y", "z".')

        new_surface = kwargs.pop("new_surface", None)
        if new_surface:
            self._surfaces.insert(index, new_surface)
            self._rebuild_state_from_surfaces()
            return

        material_pre = self._surfaces[index - 1].material_post if index > 0 else None

        if "z" in kwargs:
            z = kwargs["z"]
        else:
            thickness = kwargs.get("thickness", 0.0)
            if index == 0:
                z = -thickness
            elif index == 1:
                z = 0.0
            else:
                prev_surface = self._surfaces[index - 1]
                z = prev_surface.geometry.cs.z + prev_surface.thickness

        context = SurfaceContext(index=index, z=z, material_pre=material_pre)
        new_surface = self.surface_factory.create_surface(
            config=kwargs, context=context
        )

        self._surfaces.insert(index, new_surface)
        self._update_surface_links()

        if not self._has_absolute_z and index < len(self._surfaces) - 1:
            self._update_coordinate_systems(start_index=index + 1)

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

    def _rebuild_state_from_surfaces(self):
        self._update_surface_links()
        self._has_absolute_z = any(
            hasattr(s, "_has_absolute_z_defined") for s in self._surfaces
        )

    def remove_surface(self, index: int):
        if index == 0:
            raise ValueError("Cannot remove object surface (index 0).")
        del self._surfaces[index]
        self._rebuild_state_from_surfaces()
        if not self._has_absolute_z and index < len(self._surfaces):
            self._update_coordinate_systems(start_index=index)

    def _update_coordinate_systems(self, start_index: int):
        for i in range(start_index, len(self._surfaces)):
            prev_surface = self._surfaces[i - 1]
            new_z = prev_surface.geometry.cs.z + prev_surface.thickness
            self._surfaces[i].geometry.cs.z = be.array(new_z)

    # Keep other methods as they were to pass other tests
    def trace(self, rays, skip=0):
        self.reset()
        for surface in self.surfaces[skip:]:
            surface.trace(rays)
        return rays

    def reset(self):
        for surface in self.surfaces:
            surface.reset()

    def n(self, wavelength):
        n_values = []
        for surface in self.surfaces:
            if hasattr(surface, "material_post") and surface.material_post:
                n_values.append(be.atleast_1d(surface.material_post.n(wavelength)))
        return be.ravel(be.array(n_values))

    @property
    def radii(self):
        return be.array([surf.geometry.radius for surf in self.surfaces])

    @property
    def conic(self):
        values = []
        for surf in self.surfaces:
            try:
                values.append(surf.geometry.k)
            except AttributeError:
                values.append(0)
        return be.array(values)

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
    def uses_polarization(self):
        for surf in self.surfaces:
            if isinstance(surf.interaction_model.coating, BaseCoatingPolarized):
                return True
        return False

    def get_thickness(self, surface_number):
        t = self.positions
        return t[surface_number + 1] - t[surface_number]

    def set_fresnel_coatings(self):
        for surface in self.surfaces[1:-1]:
            if surface.material_pre != surface.material_post:
                surface.set_fresnel_coating()

    def to_dict(self) -> dict:
        return {"surfaces": [surface.to_dict() for surface in self.surfaces]}

    @classmethod
    def from_dict(cls, data: dict) -> SurfaceGroup:
        sg = cls()
        surfaces = [Surface.from_dict(s_data) for s_data in data["surfaces"]]
        sg._surfaces = surfaces
        sg._rebuild_state_from_surfaces()
        return sg

    def flip(self, start_index: int = 0, end_index: int = 0):
        n_surfaces_total = len(self._surfaces)
        if (start_index == 0 or end_index == 0) and not (
            start_index == 0 and end_index == 0
        ):
            raise RuntimeError(
                "Cannot flip object surface or image surface without flipping both"
            )
        flip_object_image_media = start_index == 0 and end_index == 0
        if flip_object_image_media:
            start_index = 1
            end_index = len(self.surfaces) - 1
        if start_index < 0:
            start_index = n_surfaces_total + start_index
        if end_index < 0:
            actual_slice_end_index = n_surfaces_total + end_index
        else:
            actual_slice_end_index = end_index
        if start_index >= actual_slice_end_index:
            self.reset()
            return
        segment_to_reverse = self._surfaces[start_index:actual_slice_end_index]
        z_positions = be.ravel(
            be.array([surf.geometry.cs.z for surf in segment_to_reverse])
        )
        segment_to_reverse.reverse()
        self._surfaces[start_index:actual_slice_end_index] = segment_to_reverse
        new_thickness = be.flip(
            be.diff(z_positions, prepend=be.array([z_positions[0]]))
        )
        new_thickness[-1] = (
            self._surfaces[actual_slice_end_index].geometry.cs.z - z_positions[-1]
        )
        new_z = (
            be.flip(be.diff(z_positions, append=be.array([z_positions[-1]]))).cumsum(0)
            + z_positions[0]
        )
        for surf, thickness, z in zip(
            segment_to_reverse, new_thickness, new_z, strict=True
        ):
            surf.flip()
            surf.geometry.cs.z = z
            surf.thickness = thickness
        if flip_object_image_media:
            self.surfaces[0].material_post, self.surfaces[-1].material_post = (
                self.surfaces[-1].material_post,
                self.surfaces[0].material_post,
            )
        self._update_surface_links()
        self.reset()
