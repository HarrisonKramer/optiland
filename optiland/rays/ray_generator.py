"""Ray Generator

This module contains the RayGenerator class, which is used to generate rays
for tracing through an optical system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.rays.polarized_rays import PolarizedRays
from optiland.rays.real_rays import RealRays


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
        vxf, vyf = self.optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)
        x0, y0, z0 = self._get_ray_origins(Hx, Hy, Px, Py, vx, vy)

        if self.optic.obj_space_telecentric:
            if self.optic.field_type == "angle":
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

        intensity = be.ones_like(x1)
        wavelength = be.ones_like(x1) * wavelength

        if self.optic.polarization == "ignore":
            if self.optic.surface_group.uses_polarization:
                raise ValueError(
                    "Polarization must be set when surfaces have "
                    "polarization-dependent coatings.",
                )
            return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)
        return PolarizedRays(x0, y0, z0, L, M, N, intensity, wavelength)

    def _get_ray_origins(self, Hx, Hy, Px, Py, vx, vy):
        """Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.

        Raises:
            ValueError: If the field type is "object_height" for an object at
                infinity.

        """
        obj = self.optic.object_surface
        max_field = self.optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        if obj.is_infinite:
            if self.optic.field_type == "object_height":
                raise ValueError(
                    'Field type cannot be "object_height" for an object at infinity.',
                )
            if self.optic.obj_space_telecentric:
                raise ValueError(
                    "Object space cannot be telecentric for an object at infinity.",
                )
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            offset = self._get_starting_z_offset()

            # x, y, z positions of ray starting points
            x = -be.tan(be.radians(field_x)) * (offset + EPL)
            y = -be.tan(be.radians(field_y)) * (offset + EPL)
            z = self.optic.surface_group.positions[1] - offset

            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            z0 = be.full_like(Px, z)
        else:
            if self.optic.field_type == "object_height":
                x = field_x
                y = field_y
                z = obj.geometry.sag(x, y) + obj.geometry.cs.z

            elif self.optic.field_type == "angle":
                EPL = self.optic.paraxial.EPL()
                z = self.optic.surface_group.positions[0]
                x = -be.tan(be.radians(field_x)) * (EPL - z)
                y = -be.tan(be.radians(field_y)) * (EPL - z)

            x0 = be.full_like(Px, x)
            y0 = be.full_like(Px, y)
            z0 = be.full_like(Px, z)

        return x0, y0, z0

    def _get_starting_z_offset(self):
        """Calculate the starting ray z-coordinate offset for systems with an
        object at infinity. This is relative to the first surface of the optic.

        This method chooses a starting point that is equivalent to the entrance
        pupil diameter of the optic.

        Returns:
            float: The z-coordinate offset relative to the first surface.

        """
        z = self.optic.surface_group.positions[1:-1]
        offset = self.optic.paraxial.EPD()
        return offset - be.min(z)
