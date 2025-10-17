"""Interaction model for diffraction

This module implements the DiffractiveInteractionModel class.

Matteo Taccola & Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.interactions.refractive_reflective_model import (
    RefractiveReflectiveModel,
)

if TYPE_CHECKING:
    # pragma: no cover
    from optiland.rays import ParaxialRays, RealRays


class DiffractiveInteractionModel(RefractiveReflectiveModel):
    """
    Interaction model for diffraction, implementing the vector grating equation.
    """

    def interact_real_rays(self, rays: RealRays) -> RealRays:
        """
        Interact with real rays using the vector diffraction grating equation.

        This method decomposes the incident ray and grating vectors into components
        parallel (tangential) and perpendicular (normal) to the surface. It then
        applies the appropriate grating equation for reflection or transmission
        to find the outgoing ray direction.

        Args:
            rays (RealRays): The incoming real rays.

        Returns:
            RealRays: The outgoing diffracted rays.
        """
        # Store incident direction for later calculations
        rays.store_incident_direction()

        # Get surface, material, and grating properties
        nx, ny, nz = self.geometry.surface_normal(rays)
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)
        fx, fy, fz = self.geometry.grating_vector(rays)
        d = self.geometry.grating_period
        m = self.geometry.grating_order

        # Align surface normal to point against the incident ray for consistent math
        nx, ny, nz, _ = rays._align_surface_normal(nx, ny, nz)

        # Correct grating period for projection effects on a curved surface.
        projection_factor = be.sqrt(fx**2 + fy**2)
        epsilon = 1e-9  # Avoid division by zero
        d_corrected = be.where(projection_factor > epsilon, d / projection_factor, d)

        # --- Vector Grating Equation Implementation ---

        # 1. Decompose incident ray vector into tangential and normal components.
        k_in_dot_n = rays.L0 * nx + rays.M0 * ny + rays.N0 * nz
        k_in_perp_L = rays.L0 - k_in_dot_n * nx
        k_in_perp_M = rays.M0 - k_in_dot_n * ny
        k_in_perp_N = rays.N0 - k_in_dot_n * nz

        # 2. Decompose grating vector into its tangential component.
        G_dot_n = fx * nx + fy * ny + fz * nz
        G_perp_L = fx - G_dot_n * nx
        G_perp_M = fy - G_dot_n * ny
        G_perp_N = fz - G_dot_n * nz

        # 3. Apply the grating equation to find the tangential component of the
        #    outgoing ray vector. The formula is different for reflection & transmission
        mu = (m * rays.w) / d_corrected  # Grating factor (m * lambda / d)

        if self.is_reflective:
            # For reflection, the tangential component is the sum of the
            # incident tangential component and the grating contribution.
            k_out_perp_L = k_in_perp_L + (mu / n1) * G_perp_L
            k_out_perp_M = k_in_perp_M + (mu / n1) * G_perp_M
            k_out_perp_N = k_in_perp_N + (mu / n1) * G_perp_N
        else:
            # For transmission, this is the generalized form of Snell's Law.
            n_ratio = n1 / n2
            k_out_perp_L = n_ratio * k_in_perp_L + (mu / n2) * G_perp_L
            k_out_perp_M = n_ratio * k_in_perp_M + (mu / n2) * G_perp_M
            k_out_perp_N = n_ratio * k_in_perp_N + (mu / n2) * G_perp_N

        # 4. Calculate the magnitude of the normal component of the outgoing ray,
        #    ensuring the final direction vector is normalized.
        k_out_perp_mag_sq = k_out_perp_L**2 + k_out_perp_M**2 + k_out_perp_N**2
        discriminant = 1.0 - k_out_perp_mag_sq

        # Handle Total Internal Reflection (TIR) for non-propagating rays.
        is_tir = discriminant < 0
        rays.clip(is_tir)
        discriminant = be.where(is_tir, be.zeros_like(discriminant), discriminant)

        k_out_para_mag = be.sqrt(discriminant)

        # 5. Combine components to get the final outgoing ray vector.
        #    The combination rule depends on the physical convention for reflection.
        if self.is_reflective:
            rays.L = -k_out_perp_L + k_out_para_mag * nx
            rays.M = -k_out_perp_M + k_out_para_mag * ny
            rays.N = -k_out_perp_N + k_out_para_mag * nz
        else:
            rays.L = k_out_perp_L + k_out_para_mag * nx
            rays.M = k_out_perp_M + k_out_para_mag * ny
            rays.N = k_out_perp_N + k_out_para_mag * nz

        # Renormalize to correct for any floating-point inaccuracies
        rays.normalize()

        # Apply coating and BSDF effects
        rays = self._apply_coating_and_bsdf(rays, nx, ny, nz)

        return rays

    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Interact with paraxial rays, causing diffraction."""
        d = self.geometry.grating_period
        m = self.geometry.grating_order

        if self.is_reflective:
            n = self.material_pre.n(rays.w)
            rays.u = -rays.u - 2 * n * rays.y / self.geometry.radius
            rays.u = rays.u + m * rays.w / d
        else:
            n1 = self.material_pre.n(rays.w)
            n2 = self.material_post.n(rays.w)
            power = (n2 - n1) / self.geometry.radius
            rays.u = (n1 / n2) * rays.u - rays.y * power / n2 - m * rays.w / (d * n2)

        return rays
