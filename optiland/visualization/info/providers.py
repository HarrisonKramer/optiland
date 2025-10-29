"""Info Providers Module

This module provides a framework for generating informational text for
different types of objects in the Optiland visualization system. It uses a
strategy pattern to allow for extensible and modular information generation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.visualization.system.lens import Lens2D
    from optiland.visualization.system.surface import Surface2D


class BaseInfoProvider:
    """Base class for information providers."""

    def get_info(self, obj: any) -> str:
        """Returns informational text for the given object."""
        raise NotImplementedError


class SurfaceInfoProvider(BaseInfoProvider):
    """Provides information for Surface2D objects."""

    def get_info(self, obj: Surface2D) -> str:
        surface = obj.surf
        surface_group = surface.parent_surface_group
        surface_index = surface_group.surfaces.index(surface)

        info = [f"Surface: {surface_index}"]
        if hasattr(surface, "comment") and surface.comment:
            info[0] += f" ({surface.comment})"
        if hasattr(surface, "geometry") and hasattr(surface.geometry, "radius"):
            info.append(f"Radius: {surface.geometry.radius:.2f}")
        if hasattr(surface, "geometry") and hasattr(surface.geometry, "conic"):
            info.append(f"Conic: {surface.geometry.conic:.2f}")

        return "\n".join(info)


class LensInfoProvider(BaseInfoProvider):
    """Provides information for Lens2D objects."""

    def get_info(self, obj: Lens2D) -> str:
        return "Lens\nMore details coming soon."


class RayBundleInfoProvider(BaseInfoProvider):
    """Provides information for ray bundles."""

    def get_info(self, obj: tuple) -> str:
        return "Ray Bundle\nMore details coming soon."


# A registry to map object types to their info providers
INFO_PROVIDER_REGISTRY = {
    "Surface2D": SurfaceInfoProvider(),
    "Lens2D": LensInfoProvider(),
    "RayBundle": RayBundleInfoProvider(),
}
