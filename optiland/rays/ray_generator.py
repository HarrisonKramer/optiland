"""Ray Generator

This module contains the RayGenerator class, which is used to generate rays
for tracing through an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

import optiland.backend as be
from optiland.rays.polarized_rays import PolarizedRays


class RayGenerator:
    """Generator class for creating rays."""

    def __init__(self, optic):
        self.optic = optic

    def generate_rays(self, Hx, Hy, Px, Py, wavelength):
        """Generates rays for tracing based on the given parameters.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            wavelength (float): Wavelength of the rays.

        Returns:
            RealRays: RealRays object containing the generated rays.

        """
        # Use the aiming context to aim the ray
        rays = self.optic.ray_aiming_context.aim_ray(self.optic, Hx, Hy, Px, Py, wavelength)

        # Apply apodization
        apodization = self.optic.apodization
        if apodization:
            rays.intensity = apodization.get_intensity(Px, Py)
        else:
            rays.intensity = be.ones_like(Px)

        rays.wavelength = be.ones_like(Px) * wavelength

        # Handle polarization
        if self.optic.polarization == "ignore":
            if self.optic.surface_group.uses_polarization:
                raise ValueError(
                    "Polarization must be set when surfaces have "
                    "polarization-dependent coatings.",
                )
            return rays
        else:
            return PolarizedRays.from_real_rays(rays)

