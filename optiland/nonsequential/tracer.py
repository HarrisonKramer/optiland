"""Non-sequential ray tracer.

Implements the core tracing loop with nearest-hit detection, interaction
dispatch (refraction/reflection), Monte Carlo path selection for coated
surfaces, and automatic TIR handling.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.physics.interaction import reflect, refract

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.nonsequential.detector import DetectorData
    from optiland.nonsequential.ray_data import NSQRayPool
    from optiland.nonsequential.surface import NSQSurface

_EPSILON = 1e-10


class NonSequentialTracer:
    """Non-sequential ray tracer using Monte Carlo path selection.

    Args:
        intensity_threshold: Minimum ray intensity before termination.
        max_interactions: Maximum number of surface interactions per trace.
    """

    def __init__(
        self,
        intensity_threshold: float = 1e-6,
        max_interactions: int = 100,
    ):
        self.intensity_threshold = intensity_threshold
        self.max_interactions = max_interactions

    def trace(
        self,
        surfaces: list[NSQSurface],
        rays: NSQRayPool,
        detectors: dict[int, DetectorData],
    ) -> None:
        """Trace rays through the scene.

        Modifies rays and detectors in-place.

        Args:
            surfaces: All surfaces in the scene.
            rays: The ray pool to trace.
            detectors: Mapping from detector surface_id to DetectorData.
        """
        interactions = 0

        while rays.n_active > 0 and interactions < self.max_interactions:
            # 1. Find nearest intersection for each active ray
            distances, nearest_ids = self._find_nearest_intersection(
                rays,
                surfaces,
            )

            # 2. Deactivate rays that hit nothing
            no_hit = nearest_ids == -1
            rays.deactivate(no_hit)
            if rays.n_active == 0:
                break

            # 3. Propagate rays to intersection points
            rays.propagate(distances)

            # 4. Process each hit surface
            for surface in surfaces:
                hit_mask = (nearest_ids == surface.surface_id) & rays.active
                if not be.any(hit_mask):
                    continue

                # Record path point
                rays.record_path_point(nearest_ids)

                # Record on detector
                if surface.is_detector and surface.surface_id in detectors:
                    detectors[surface.surface_id].record(rays, hit_mask)

                # Compute surface normal at hit points
                nx, ny, nz = surface.surface_normal(rays)

                # Determine which side ray approaches from
                dot = rays.L * nx + rays.M * ny + rays.N * nz

                # Flip normal to face incoming ray (make dot negative)
                flip = dot > 0
                nx = be.where(flip, -nx, nx)
                ny = be.where(flip, -ny, ny)
                nz = be.where(flip, -nz, nz)
                dot = be.where(flip, -dot, dot)

                # Aligned normal now points toward incoming ray, dot < 0
                # For physics kernels we need positive dot (absolute value)
                nx_aligned = -nx
                ny_aligned = -ny
                nz_aligned = -nz
                dot_abs = -dot  # = abs(original dot)

                # Determine materials based on approach side
                # front = normal aligns with approach → flip was True
                front_side = flip

                n1_vals = be.where(
                    front_side,
                    self._get_n(
                        surface.material_front,
                        rays.wavelength,
                        hit_mask,
                    ),
                    self._get_n(
                        surface.material_back,
                        rays.wavelength,
                        hit_mask,
                    ),
                )
                n2_vals = be.where(
                    front_side,
                    self._get_n(
                        surface.material_back,
                        rays.wavelength,
                        hit_mask,
                    ),
                    self._get_n(
                        surface.material_front,
                        rays.wavelength,
                        hit_mask,
                    ),
                )

                if surface.is_reflective:
                    L_new, M_new, N_new = reflect(
                        rays.L,
                        rays.M,
                        rays.N,
                        nx_aligned,
                        ny_aligned,
                        nz_aligned,
                    )
                    rays.update_directions(L_new, M_new, N_new, hit_mask)
                else:
                    # Attempt refraction
                    L_ref, M_ref, N_ref, tir_mask = refract(
                        rays.L,
                        rays.M,
                        rays.N,
                        nx_aligned,
                        ny_aligned,
                        nz_aligned,
                        n1_vals,
                        n2_vals,
                    )

                    # Auto-reflect on TIR
                    tir_and_hit = tir_mask & hit_mask
                    if be.any(tir_and_hit):
                        L_refl, M_refl, N_refl = reflect(
                            rays.L,
                            rays.M,
                            rays.N,
                            nx_aligned,
                            ny_aligned,
                            nz_aligned,
                        )
                        L_ref = be.where(tir_mask, L_refl, L_ref)
                        M_ref = be.where(tir_mask, M_refl, M_ref)
                        N_ref = be.where(tir_mask, N_refl, N_ref)

                    # Monte Carlo for coating with partial R/T
                    if surface.coating is not None:
                        L_ref, M_ref, N_ref = self._apply_coating_mc(
                            surface,
                            rays,
                            hit_mask,
                            nx_aligned,
                            ny_aligned,
                            nz_aligned,
                            L_ref,
                            M_ref,
                            N_ref,
                            dot_abs,
                            n1_vals,
                            n2_vals,
                        )

                    rays.update_directions(L_ref, M_ref, N_ref, hit_mask)

                # Update last surface ID
                rays.last_surface_id = be.where(
                    hit_mask,
                    be.full_like(rays.last_surface_id, surface.surface_id),
                    rays.last_surface_id,
                )

            # 5. Apply epsilon offset to avoid self-intersection
            eps = be.full(len(rays.x), _EPSILON)
            rays.propagate(eps)

            # 6. Kill low-intensity rays
            low_energy = rays.intensity < self.intensity_threshold
            rays.deactivate(low_energy)

            interactions += 1

    def _find_nearest_intersection(
        self,
        rays: NSQRayPool,
        surfaces: list[NSQSurface],
    ) -> tuple[BEArray, BEArray]:
        """Find the nearest surface intersection for each active ray.

        Args:
            rays: The ray pool.
            surfaces: All surfaces in the scene.

        Returns:
            min_distances: Nearest intersection distance per ray.
            nearest_surface_ids: Surface ID of nearest hit (-1 if no hit).
        """
        n = len(rays.x)
        min_distances = be.full(n, float("inf"))
        nearest_ids = be.full(n, -1.0)

        for surface in surfaces:
            # Skip self-intersection
            is_last = rays.last_surface_id == surface.surface_id

            distances, hit_mask = surface.intersect(rays)

            # Exclude self-intersections and inactive rays
            valid = hit_mask & rays.active & ~is_last

            closer = valid & (distances < min_distances)
            min_distances = be.where(closer, distances, min_distances)
            nearest_ids = be.where(
                closer,
                be.full_like(nearest_ids, surface.surface_id),
                nearest_ids,
            )

        return min_distances, nearest_ids

    def _get_n(
        self,
        material: object,
        wavelengths: BEArray,
        mask: BEArray,
    ) -> BEArray:
        """Get refractive index array for all rays from a material.

        Args:
            material: A BaseMaterial instance.
            wavelengths: Per-ray wavelength array.
            mask: Hit mask (used for shape reference only).

        Returns:
            Array of refractive indices, one per ray.
        """
        # Get unique wavelengths to minimize material calls
        unique_w = be.unique(wavelengths)
        result = be.ones_like(wavelengths)
        for w in unique_w:
            w_val = float(w)
            n_val = float(material.n(w_val))
            w_mask = wavelengths == w
            result = be.where(w_mask, be.array(n_val), result)
        return result

    def _apply_coating_mc(
        self,
        surface: NSQSurface,
        rays: NSQRayPool,
        hit_mask: BEArray,
        nx: BEArray,
        ny: BEArray,
        nz: BEArray,
        L_refracted: BEArray,
        M_refracted: BEArray,
        N_refracted: BEArray,
        dot_abs: BEArray,
        n1: BEArray,
        n2: BEArray,
    ) -> tuple[BEArray, BEArray, BEArray]:
        """Apply Monte Carlo coating interaction.

        For surfaces with coatings that have both R > 0 and T > 0,
        randomly select reflection or transmission per ray.

        Args:
            surface: The surface with coating.
            rays: The ray pool.
            hit_mask: Boolean mask of rays hitting this surface.
            nx, ny, nz: Aligned surface normals.
            L_refracted, M_refracted, N_refracted: Refracted directions.
            dot_abs: Absolute dot product of incident direction and normal.
            n1, n2: Refractive indices.

        Returns:
            Updated direction cosines (L, M, N).
        """
        coating = surface.coating

        # Get R and T from coating
        R = getattr(coating, "reflectance", 0.0)
        T = getattr(coating, "transmittance", 1.0)

        if R <= 0:
            # Pure transmission, just apply coating factor
            rays.apply_intensity(
                be.full_like(rays.intensity, T),
                hit_mask,
            )
            return L_refracted, M_refracted, N_refracted

        R_plus_T = R + T
        if R_plus_T <= 0:
            return L_refracted, M_refracted, N_refracted

        # Draw random numbers for Monte Carlo selection
        rand = be.array(
            np.random.uniform(0, 1, size=len(rays.x)),
        )
        choose_reflect = (rand < R / R_plus_T) & hit_mask

        # Reflect those chosen for reflection
        L_refl, M_refl, N_refl = reflect(
            rays.L,
            rays.M,
            rays.N,
            nx,
            ny,
            nz,
        )

        L_out = be.where(choose_reflect, L_refl, L_refracted)
        M_out = be.where(choose_reflect, M_refl, M_refracted)
        N_out = be.where(choose_reflect, N_refl, N_refracted)

        # Energy conservation: scale intensity by (R + T)
        rays.apply_intensity(
            be.full_like(rays.intensity, R_plus_T),
            hit_mask,
        )

        return L_out, M_out, N_out
