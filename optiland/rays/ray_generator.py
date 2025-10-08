"""Ray Generator

This module contains the RayGenerator class, which is used to generate rays
for tracing through an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.fields.field_types import AngleField
from optiland.rays.polarized_rays import PolarizedRays
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray


class RayGenerator:
    """Generator class for creating rays."""

    def __init__(self, optic):
        self.optic = optic

    def generate_rays(
        self,
        Hx: float,
        Hy: float,
        Px: ScalarOrArray,
        Py: ScalarOrArray,
        wavelength: ScalarOrArray,
    ) -> RealRays:
        """Generates rays for tracing based on the given parameters.

        Args:
            Hx: Normalized x field coordinate.
            Hy: Normalized y field coordinate.
            Px: x-coordinate of the pupil point.
            Py: y-coordinate of the pupil point.
            wavelength: Wavelength of the rays.

        Returns:
            RealRays object containing the generated rays.

        """
        vxf, vyf = self.optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)
        x0, y0, z0 = self.optic.field_definition.get_ray_origins(
            self.optic, Hx, Hy, Px, Py, vx, vy
        )

        if self.optic.obj_space_telecentric:
            if isinstance(self.optic.field_definition, AngleField):
                raise ValueError(
                    'Field type cannot be "angle" for telecentric object space.',
                )
            if self.optic.aperture.ap_type == "EPD":
                raise ValueError(
                    'Aperture type cannot be "EPD" for telecentric object space.',
                )
            if self.optic.aperture.ap_type == "imageFNO":
                raise ValueError(
                    'Aperture type cannot be "imageFNO" for telecentric object space.',
                )

            sin = self.optic.aperture.value
            z = be.sqrt(1 - sin**2) / sin + z0
            z1 = be.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            z1 = be.full_like(Px, EPL)

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        apodization = self.optic.apodization
        if apodization:
            intensity = apodization.get_intensity(Px, Py)
        else:
            intensity = be.ones_like(Px)

        wavelength = be.ones_like(x1) * wavelength

        if self.optic.polarization == "ignore":
            if self.optic.surface_group.uses_polarization:
                raise ValueError(
                    "Polarization must be set when surfaces have "
                    "polarization-dependent coatings.",
                )
            return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)
        return PolarizedRays(x0, y0, z0, L, M, N, intensity, wavelength)
