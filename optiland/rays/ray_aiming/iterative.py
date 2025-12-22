"""Iterative Ray Aiming Module

This module implements the iterative ray aiming algorithm, which uses a
Newton-Raphson-like method to aim rays at the stop surface for aberrated
systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.rays import RealRays
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer
from optiland.rays.ray_aiming.registry import register_aimer


@register_aimer("iterative")
class IterativeRayAimer(BaseRayAimer):
    """Iterative ray aiming strategy.

    This aimer uses the paraxial result as a starting guess and then iteratively
    adjusts the launch coordinates/direction to ensure the rays intersect the
    stop surface at the desired normalized pupil coordinates.

    Args:
        optic: The optical system to aim rays for.
        max_iter: Maximum number of iterations. Defaults to 10.
        tol: Convergence tolerance for the pupil error. Defaults to 1e-6.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, optic, max_iter: int = 20, tol: float = 1e-8, **kwargs):
        super().__init__(optic, **kwargs)
        self.max_iter = max_iter
        self.tol = tol
        # Internal paraxial aimer for initial guess
        self._paraxial_aimer = ParaxialRayAimer(optic)

    def aim_rays(
        self,
        fields: tuple,
        wavelengths: float,
        pupil_coords: tuple,
    ) -> tuple:
        """Calculate ray starting coordinates using iterative aiming."""
        # 1. Get initial guess from paraxial aimer
        x, y, z, L, M, N = self._paraxial_aimer.aim_rays(
            fields, wavelengths, pupil_coords
        )

        Px, Py = pupil_coords
        is_infinite = (
            self.optic.object_surface.is_infinite
            if self.optic.object_surface
            else False
        )
        stop_index = self.optic.surface_group.stop_index
        stop_surface = self.optic.surface_group.surfaces[stop_index]

        # Determine the physical target radius on the stop surface.
        # Ideally, we use the explicit aperture dimensions if available.

        if stop_surface.aperture:
            aperture = stop_surface.aperture
            if hasattr(aperture, "r_max"):
                rx = ry = aperture.r_max
            elif hasattr(aperture, "x_max") and hasattr(aperture, "y_max"):
                rx = aperture.x_max
                ry = aperture.y_max
            else:
                # Fallback using extent
                min_x, max_x, min_y, max_y = aperture.extent
                rx = max_x
                ry = max_y
        else:
            # Fallback to paraxial approximation if no physical aperture is defined.
            rx = ry = self.optic.paraxial.EPD() / 2.0
            if stop_surface.semi_aperture:
                rx = ry = stop_surface.semi_aperture

        tx = Px * rx
        ty = Py * ry

        for _ in range(self.max_iter):
            rays = RealRays(
                x, y, z, L, M, N, intensity=be.ones_like(x), wavelength=wavelengths
            )

            # Trace rays from the object (or first surface) up to the stop surface.
            for i in range(stop_index + 1):
                surf = self.optic.surface_group.surfaces[i]
                surf.trace(rays)

            # Calculate error at stop surface relative to target pupil coordinates.
            ex = rays.x - tx
            ey = rays.y - ty

            # Check convergence (max error across batch)
            error_mag = be.sqrt(ex**2 + ey**2)
            max_error = be.max(error_mag)

            if max_error < self.tol:
                break

            # Update ray launch parameters based on the error.
            if is_infinite:
                # For infinite conjugates, adjust the launch position (x, y).
                # The beam angle (L, M, N) remains fixed.
                x = x - ex
                y = y - ey
            else:
                # For finite conjugates, adjust the launch direction (L, M).
                # The launch position (x, y, z) remains fixed.
                # Approximate distance from launch z to current ray z at stop.
                dist = be.abs(rays.z - z)
                dist = be.where(dist < 1e-3, 1.0, dist)  # Prevent division by zero

                L = L - (ex / dist)
                M = M - (ey / dist)

                # Renormalize N to ensure L^2 + M^2 + N^2 = 1.
                sq_sum = L**2 + M**2
                sq_sum = be.where(sq_sum > 1.0, 1.0, sq_sum)
                N_mag = be.sqrt(1.0 - sq_sum)

                # Preserve the original sign of N.
                N = be.where(N >= 0, N_mag, -N_mag)

        return x, y, z, L, M, N
