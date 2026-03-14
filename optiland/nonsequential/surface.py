"""Non-sequential surface.

A clean adapter that holds geometry + materials + interaction without
sequential assumptions like thickness or surface chaining.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays import RealRays

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.coatings import BaseCoating
    from optiland.geometries.base import BaseGeometry
    from optiland.materials import BaseMaterial
    from optiland.nonsequential.ray_data import NSQRayPool


class NSQSurface:
    """A surface in a non-sequential scene.

    Unlike sequential Surface, this has no thickness, no previous_surface
    linking, and explicitly stores materials on both sides.

    Args:
        geometry: The surface geometry (spherical, planar, aspheric, etc.).
        material_front: Material on the front side of the surface.
        material_back: Material on the back side of the surface.
        coating: Optional coating for partial R/T.
        is_reflective: Whether this surface is purely reflective.
        is_detector: Whether to record ray hits on this surface.
        label: Human-readable label for the surface.
    """

    def __init__(
        self,
        geometry: BaseGeometry,
        material_front: BaseMaterial,
        material_back: BaseMaterial,
        coating: BaseCoating | None = None,
        is_reflective: bool = False,
        is_detector: bool = False,
        label: str = "",
    ):
        self.geometry = geometry
        self.material_front = material_front
        self.material_back = material_back
        self.coating = coating
        self.is_reflective = is_reflective
        self.is_detector = is_detector
        self.label = label
        self._surface_id: int = -1

    @property
    def surface_id(self) -> int:
        """Unique ID assigned when added to a scene."""
        return self._surface_id

    @surface_id.setter
    def surface_id(self, value: int) -> None:
        self._surface_id = value

    def intersect(self, rays: NSQRayPool) -> tuple[BEArray, BEArray]:
        """Find intersection distances for all rays.

        Creates a temporary RealRays copy in local coordinates to use
        the existing geometry.distance() method, without modifying the
        original ray pool.

        Args:
            rays: The ray pool to test against this surface.

        Returns:
            distances: Propagation distance to surface (inf if no hit).
            hit_mask: Boolean mask of rays that hit this surface.
        """
        # Create temporary RealRays for geometry interface
        temp_rays = RealRays(
            be.copy(rays.x),
            be.copy(rays.y),
            be.copy(rays.z),
            be.copy(rays.L),
            be.copy(rays.M),
            be.copy(rays.N),
            be.copy(rays.intensity),
            be.copy(rays.wavelength),
        )

        # Transform to local coordinates
        self.geometry.localize(temp_rays)

        # Compute distances in local coordinates
        distances = self.geometry.distance(temp_rays)

        # Post-process: negative/NaN distances → inf (no valid hit)
        inf_val = be.array(float("inf"))
        with be.errstate(invalid="ignore"):
            distances = be.where(
                be.isnan(distances) | (distances <= 0),
                inf_val,
                distances,
            )

        hit_mask = ~be.isinf(distances)
        return distances, hit_mask

    def surface_normal(self, rays: NSQRayPool) -> tuple[BEArray, BEArray, BEArray]:
        """Compute surface normals at current ray positions.

        Localizes ray positions, computes normals, then returns normals
        in global coordinates.

        Args:
            rays: Ray pool positioned at surface intersection points.

        Returns:
            nx, ny, nz: Surface normal components in global coordinates.
        """
        # Create temporary RealRays at the intersection positions
        temp_rays = RealRays(
            be.copy(rays.x),
            be.copy(rays.y),
            be.copy(rays.z),
            be.copy(rays.L),
            be.copy(rays.M),
            be.copy(rays.N),
            be.copy(rays.intensity),
            be.copy(rays.wavelength),
        )

        # Transform to local coordinates
        self.geometry.localize(temp_rays)

        # Get normals in local coordinates
        nx, ny, nz = self.geometry.surface_normal(temp_rays)

        # Transform normals back to global coordinates by using a
        # direction-only transform: create rays at origin with normal as
        # direction, globalize, then extract direction.
        normal_rays = RealRays(
            be.zeros_like(nx),
            be.zeros_like(ny),
            be.zeros_like(nz),
            nx,
            ny,
            nz,
            be.ones_like(nx),
            be.ones_like(nx),
        )
        self.geometry.globalize(normal_rays)

        # Extract only the direction (normals are direction-only)
        mag = be.sqrt(normal_rays.L**2 + normal_rays.M**2 + normal_rays.N**2)
        nx_g = normal_rays.L / mag
        ny_g = normal_rays.M / mag
        nz_g = normal_rays.N / mag

        return nx_g, ny_g, nz_g

    def determine_side(
        self,
        rays: NSQRayPool,
        nx: BEArray,
        ny: BEArray,
        nz: BEArray,
    ) -> BEArray:
        """Determine which side each ray approaches from.

        Args:
            rays: The ray pool.
            nx: x-component of surface normals.
            ny: y-component of surface normals.
            nz: z-component of surface normals.

        Returns:
            Boolean mask: True = front side, False = back side.
        """
        dot = rays.L * nx + rays.M * ny + rays.N * nz
        # If dot > 0, ray direction aligns with normal → front side
        # Convention: normal points from front to back
        return dot > 0
