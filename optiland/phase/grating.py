"""Grating Phase

This module defines the `GratingPhase` class, which represents a phase
function for a diffraction grating.

The `GratingPhase` class calculates the change in the direction of a ray and
the optical path difference (OPD) introduced by a diffraction grating. It
supports different grating orders and orientations.

The implementation is based on the vector formulation of the grating equation,
which allows for the calculation of the diffracted ray direction for any
incident ray and grating orientation.

Hhsoj, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.phase.base import BasePhase

if TYPE_CHECKING:
    from optiland.rays import RealRays
    from optiland._types import BEArray


class GratingPhase(BasePhase):
    """Represents a phase function for a diffraction grating.

    Args:
        period (float): The period of the grating in lines/μm.
        order (int): The diffraction order.
        gx (float, optional): The x-component of the grating vector.
            Defaults to 1.
        gy (float, optional): The y-component of the grating vector.
            Defaults to 0.
        gz (float, optional): The z-component of the grating vector.
            Defaults to 0.

    """

    def __init__(
        self,
        period: float = 1.0,
        order: int = 1,
        gx: float = 1.0,
        gy: float = 0.0,
        gz: float = 0.0,
    ):
        self.period = be.array(period)
        self.order = be.array(order)
        self.gx = gx
        self.gy = gy
        self.gz = gz

    def phase_calc(
        self,
        rays: RealRays,
        nx: BEArray,
        ny: BEArray,
        nz: BEArray,
        n1: BEArray,
        n2: BEArray,
    ) -> tuple[BEArray, BEArray, BEArray, BEArray]:
        """Calculates the effect of the grating phase function on the rays.

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
        spacing = 1 / self.period
        nx = -nx
        ny = -ny
        nz = -nz

        # cross product of n x g
        tx = ny * self.gz - nz * self.gy
        ty = nz * self.gx - nx * self.gz
        tz = nx * self.gy - ny * self.gx
        mag = be.sqrt(tx * tx + ty * ty + tz * tz)

        tx = be.where(mag <= 0, 0, tx / mag)
        ty = be.where(mag <= 0, 0, ty / mag)
        tz = be.where(mag <= 0, 0, tz / mag)

        Kx = (2 * be.pi / spacing) * tx
        Ky = (2 * be.pi / spacing) * ty
        Kz = (2 * be.pi / spacing) * tz

        # define parameters
        dx, dy, dz = rays.L, rays.M, rays.N

        wavelength = rays.w
        k_mag = 2 * be.pi / wavelength
        kix = k_mag * dx
        kiy = k_mag * dy
        kiz = k_mag * dz

        dot_kn = kix * nx + kiy * ny + kiz * nz
        kpx = kix - dot_kn * nx
        kpy = kiy - dot_kn * ny
        kpz = kiz - dot_kn * nz

        m = self.order

        kdx = kpx + m * Kx
        kdy = kpy + m * Ky
        kdz = kpz + m * Kz

        kp2 = kdx**2 + kdy**2 + kdz**2

        dk_mag2_kp2 = k_mag**2 - kp2
        if be.any(dk_mag2_kp2 < 0):
            raise ValueError("Total internal reflection due to phase.")

        k_perp_mag = be.sqrt(dk_mag2_kp2)

        kfx = kdx + k_perp_mag * nx
        kfy = kdy + k_perp_mag * ny
        kfz = kdz + k_perp_mag * nz

        uk = be.sqrt(kfx**2 + kfy**2 + kfz**2)

        L = kfx / uk
        M = kfy / uk
        N = kfz / uk

        dot_knn = dx * nx + dy * ny + dz * nz
        sin_in = be.sqrt(1 - dot_knn**2)
        dot_kfn = L * nx + M * ny + N * nz
        sin_out = be.sqrt(1 - dot_kfn**2)
        d = 1 / self.period
        opd = d * (n1 * sin_in + n2 * sin_out)

        return L, M, N, opd

    def efficiency(self, rays: RealRays) -> BEArray:
        """Calculates the diffraction efficiency of the grating.

        For now, this returns an ideal efficiency of 1.0.

        Args:
            rays (RealRays): The rays incident on the surface.

        Returns:
            BEArray: The diffraction efficiency for each ray.

        """
        return be.ones_like(rays.x)

    def to_dict(self) -> dict:
        """Converts the GratingPhase object to a dictionary."""
        phase_dict = super().to_dict()
        phase_dict.update(
            {
                "period": float(self.period),
                "order": int(self.order),
                "gx": self.gx,
                "gy": self.gy,
                "gz": self.gz,
            }
        )
        return phase_dict

    @classmethod
    def from_dict(cls, data: dict) -> "GratingPhase":
        """Creates a GratingPhase object from a dictionary."""
        return cls(
            period=data["period"],
            order=data["order"],
            gx=data.get("gx", 1.0),
            gy=data.get("gy", 0.0),
            gz=data.get("gz", 0.0),
        )
