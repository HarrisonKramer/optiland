"""Paraxial Ray Aiming Strategy.

This module provides a ray aiming strategy that uses paraxial optics to aim
rays.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.aiming.strategies.base import RayAimingStrategy
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland.optic.optic import Optic


class ParaxialAimingStrategy(RayAimingStrategy):
    """A ray aiming strategy that uses paraxial optics to aim rays."""

    def aim(
        self,
        optic: Optic,
        Hx: ArrayLike,
        Hy: ArrayLike,
        Px: ArrayLike,
        Py: ArrayLike,
        wavelength: float,
    ):
        """Given an optic, normalized field and pupil coordinates and wavelength,
        return the ray.

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
        vxf, vyf = optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)
        x0, y0, z0 = self._get_ray_origins(optic, Hx, Hy, Px, Py, vx, vy)

        if optic.obj_space_telecentric:
            if optic.field_type == "angle":
                raise ValueError(
                    'Field type cannot be "angle" for telecentric object space.',
                )
            if optic.aperture.ap_type == "EPD":
                raise ValueError(
                    'Aperture type cannot be "EPD" for telecentric object space.',
                )
            if optic.aperture.ap_type == "imageFNO":
                raise ValueError(
                    'Aperture type cannot be "imageFNO" for telecentric object space.',
                )

            sin = optic.aperture.value
            z = be.sqrt(1 - sin**2) / sin + z0
            z1 = be.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()

            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            z1 = be.full_like(Px, EPL)

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        wavelength = be.ones_like(x0) * wavelength
        intensity = be.ones_like(x0)

        return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)

    def _get_ray_origins(
        self,
        optic: Optic,
        Hx: ArrayLike,
        Hy: ArrayLike,
        Px: ArrayLike,
        Py: ArrayLike,
        vx: ArrayLike,
        vy: ArrayLike,
    ):
        """Calculate the initial positions for rays originating at the object.

        Args:
            optic: The optic to calculate the ray origins for.
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            vx: The x vignetting factor(s).
            vy: The y vignetting factor(s).

        Returns:
            A tuple containing the x, y, and z coordinates of the ray origins.
        """
        obj = optic.object_surface
        max_field = optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        if obj.is_infinite:
            if optic.field_type == "object_height":
                raise ValueError(
                    'Field type cannot be "object_height" for an object at infinity.',
                )
            if optic.obj_space_telecentric:
                raise ValueError(
                    "Object space cannot be telecentric for an object at infinity.",
                )
            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()

            offset = self._get_starting_z_offset(optic)

            # x, y, z positions of ray starting points
            x = -be.tan(be.radians(field_x)) * (offset + EPL)
            y = -be.tan(be.radians(field_y)) * (offset + EPL)
            z = optic.surface_group.positions[1] - offset

            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            z0 = be.full_like(Px, z)
        else:
            if optic.field_type == "object_height":
                x0 = be.array(field_x)
                y0 = be.array(field_y)
                z0 = obj.geometry.sag(x0, y0) + obj.geometry.cs.z

            elif optic.field_type == "angle":
                EPL = optic.paraxial.EPL()
                z0 = optic.surface_group.positions[0]
                x0 = -be.tan(be.radians(field_x)) * (EPL - z0)
                y0 = -be.tan(be.radians(field_y)) * (EPL - z0)

            if be.size(x0) == 1:
                x0 = be.full_like(Px, x0)
            if be.size(y0) == 1:
                y0 = be.full_like(Px, y0)
            if be.size(z0) == 1:
                z0 = be.full_like(Px, z0)

        return x0, y0, z0

    def _get_starting_z_offset(self, optic: Optic) -> float:
        """Calculate the starting ray z-coordinate offset for systems with an
        object at infinity. This is relative to the first surface of the optic.

        This method chooses a starting point that is equivalent to the entrance
        pupil diameter of the optic.

        Args:
            optic: The optic to calculate the offset for.

        Returns:
            The z-coordinate offset relative to the first surface.
        """
        z = optic.surface_group.positions[1:-1]
        offset = optic.paraxial.EPD()
        if len(z) > 0:
            return offset - be.min(z)
        return offset
