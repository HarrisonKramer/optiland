"""Types for Surface Factories

This module defines common data structures used by the surface factories.

Kramer Harrison, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial


@dataclass
class SurfaceContext:
    """A data container for the contextual state needed to create a surface.

    Attributes:
        index: The index of the surface to be created.
        z: The absolute z-position for the new surface's coordinate system.
        material_pre: The material preceding the new surface.
    """

    index: int
    z: float
    material_pre: BaseMaterial | None
