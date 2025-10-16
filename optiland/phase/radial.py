"""Radial Phase

This module defines the `RadialPhase` class, which represents a radially
symmetric phase function.

The `RadialPhase` class calculates the change in the direction of a ray and
the optical path difference (OPD) introduced by a radially symmetric phase
profile. The phase profile is defined by a series of coefficients.

The implementation is based on the calculation of the phase gradient, which is
used to determine the change in the ray's direction.

Hhsoj, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import optiland.backend as be
from optiland.phase.base import BasePhase

if TYPE_CHECKING:
    from optiland.rays import RealRays
    from optiland._types import BEArray


class RadialPhase(BasePhase):
    """Represents a radially symmetric phase function.

    Args:
        order (int, optional): The diffraction order. Defaults to 1.
        coefficients (List[float], optional): A list of coefficients that
            define the radial phase profile. Defaults to None.

    """

    def __init__(self, order: int = 1, coefficients: List[float] = None):
        if coefficients is None:
            coefficients = []
        self.order = be.array(order)
        self.coefficients = be.array(coefficients)

    def phase_calc(
        self,
        rays: RealRays,
        nx: BEArray,
        ny: BEArray,
        nz: BEArray,
        n1: BEArray,
        n2: BEArray,
    ) -> tuple[BEArray, BEArray, BEArray, BEArray]:
        """Calculates the effect of the radial phase function on the rays.

        Args:
            rays (RealRays): The rays incident on the surface.
            nx (BEArray): The x-component of the surface normal.
            ny (BEArray): The y-component of the surface normal.
            nz (BEArray): The z-component of the surface normal.
            n1 (BEArray): The refractive index of the medium before the
                surface.
            n2 (BEArray): The refractive index of the medium after the
                surface.

        Returns:
            A tuple containing the new x, y, and z direction cosines (L, M, N)
            and the optical path difference (OPD) to be added to the rays.

        """
        nx = -1 * nx
        ny = -1 * ny
        nz = -1 * nz

        m = self.order
        r = be.sqrt(rays.x**2 + rays.y**2)

        dphi_dr = sum(
            2 * i * a * r ** (2 * i - 1)
            for i, a in enumerate(self.coefficients, start=1)
        )
        with be.errstate(divide="ignore", invalid="ignore"):
            dphi_dx = be.where(r != 0, dphi_dr * rays.x / r, 0.0)
            dphi_dy = be.where(r != 0, dphi_dr * rays.y / r, 0.0)

        wavelength = rays.w
        wavelength_ratio = wavelength

        incident_cos_angle = rays.L * nx + rays.M * ny + rays.N * nz

        b = incident_cos_angle + m * (nx * dphi_dx + ny * dphi_dy)
        c = wavelength_ratio * (
            wavelength_ratio * (dphi_dx**2 + dphi_dy**2) / 2
            + m * (rays.L * dphi_dx + rays.M * dphi_dy)
        )

        discriminant = b**2 - 2 * c
        if be.any(discriminant < 0):
            raise ValueError("Total internal reflection due to phase.")

        Q = -b + rays.N * be.sqrt(discriminant)

        L = rays.L + m * wavelength_ratio * dphi_dx + Q * nx
        M = rays.M + m * wavelength_ratio * dphi_dy + Q * ny
        N = rays.N + Q * nz

        out_mag = be.sqrt(L**2 + M**2 + N**2)
        L /= out_mag
        M /= out_mag
        N /= out_mag

        opd = m * sum(a * r ** (2 * i) for i, a in enumerate(self.coefficients, start=1))

        return L, M, N, opd

    def efficiency(self, rays: RealRays) -> BEArray:
        """Calculates the diffraction efficiency of the radial phase function.

        For now, this returns an ideal efficiency of 1.0.

        Args:
            rays (RealRays): The rays incident on the surface.

        Returns:
            BEArray: The diffraction efficiency for each ray.

        """
        return be.ones_like(rays.x)

    def to_dict(self) -> dict:
        """Converts the RadialPhase object to a dictionary."""
        phase_dict = super().to_dict()
        phase_dict.update(
            {
                "order": int(self.order),
                "coefficients": self.coefficients.tolist(),
            }
        )
        return phase_dict

    @classmethod
    def from_dict(cls, data: dict) -> "RadialPhase":
        """Creates a RadialPhase object from a dictionary."""
        return cls(
            order=data.get("order", 1),
            coefficients=data.get("coefficients"),
        )
