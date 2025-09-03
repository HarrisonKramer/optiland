"""Fallback Ray Aiming Strategy.

This module provides a ray aiming strategy that falls back to a secondary
strategy if the primary fails.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.aiming.strategies.base import RayAimingStrategy
from optiland.aiming.strategies.iterative import IterativeAimingStrategy
from optiland.aiming.strategies.paraxial import ParaxialAimingStrategy
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland.optic.optic import Optic


class FallbackAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that falls back to a secondary strategy if the
    primary fails.
    """

    def __init__(
        self,
        primary: RayAimingStrategy | None = None,
        secondary: RayAimingStrategy | None = None,
        pupil_error_threshold: float = 1e-2,
    ):
        """Initializes the FallbackAimingStrategy.

        Args:
            primary: The primary strategy to use. If None,
                IterativeAimingStrategy is used.
            secondary: The secondary strategy to use. If None,
                ParaxialAimingStrategy is used.
            pupil_error_threshold: The pupil error threshold to use for
                falling back to the secondary strategy.
        """
        self.primary = primary if primary is not None else IterativeAimingStrategy()
        self.secondary = (
            secondary if secondary is not None else ParaxialAimingStrategy()
        )
        self.pupil_error_threshold = pupil_error_threshold

    def aim_ray(
        self,
        optic: Optic,
        Hx: ArrayLike,
        Hy: ArrayLike,
        Px: ArrayLike,
        Py: ArrayLike,
        wavelength: float,
    ):
        """Aims a ray using a primary strategy and falling back to a secondary
        strategy if the primary fails.

        Args:
            optic: The optic to aim the ray for.
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength: The wavelength of the ray in microns.

        Returns:
            The aimed ray(s).
        """
        try:
            primary_rays = self.primary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

            # Trace the rays to check for failure and pupil error
            rays_to_trace = RealRays.from_other(primary_rays)
            optic.surface_group.trace(rays_to_trace)

            if be.any(rays_to_trace.fail):
                return self.secondary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

            # Check pupil error
            stop_idx = optic.surface_group.stop_index
            pupil_x = optic.surface_group.x[stop_idx]
            pupil_y = optic.surface_group.y[stop_idx]

            aperture = optic.surface_group.surfaces[stop_idx].aperture
            if not (aperture and hasattr(aperture, "r_max") and aperture.r_max > 0):
                return primary_rays

            stop_radius = aperture.r_max
            actual_px = pupil_x / stop_radius
            actual_py = pupil_y / stop_radius

            error = be.sqrt(
                (actual_px - be.array(Px)) ** 2 + (actual_py - be.array(Py)) ** 2
            )

            if be.any(error > self.pupil_error_threshold):
                return self.secondary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

            return primary_rays
        except Exception:
            return self.secondary.aim_ray(optic, Hx, Hy, Px, Py, wavelength)
