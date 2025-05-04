"""Paraxial Ray Tracer Module

This module contains the ParaxialRayTracer class, which is responsible for tracing
paraxial rays through an optical system.

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.rays.paraxial_rays import ParaxialRays
from optiland.surfaces import ObjectSurface


class ParaxialRayTracer:
    """Class to trace paraxial rays through an optical system

    Args:
        optic (Optic): The optical system to be traced.
    """

    def __init__(self, optic):
        self.optic = optic

    def trace(self, Hy, Py, wavelength):
        """Trace paraxial ray through the optical system based on specified field
        and pupil coordinates.

        Args:
            Hy (float): Normalized field coordinate.
            Py (float): Normalized pupil coordinate.
            wavelength (float): Wavelength of the light.

        """
        EPL = self.optic.paraxial.EPL()
        EPD = self.optic.paraxial.EPD()

        y1 = Py * EPD / 2

        y0, z0 = self._get_object_position(Hy, y1, EPL)
        u0 = (y1 - y0) / (EPL - z0)
        rays = ParaxialRays(y0, u0, z0, wavelength)

        self.optic.surface_group.trace(rays)

    def trace_generic(self, y, u, z, wavelength, reverse=False, skip=0):
        """
        Trace generically-defined paraxial rays through the optical system.

        Args:
            y (float or array-like): The initial height(s) of the rays.
            u (float or array-like): The initial slope(s) of the rays.
            z (float or array-like): The initial axial position(s) of the rays.
            wavelength (float): The wavelength of the rays.
            reverse (bool, optional): If True, trace the rays in reverse
                direction. Defaults to False.
            skip (int, optional): The number of surfaces to skip during
                tracing. Defaults to 0.

        Returns:
            tuple: A tuple containing the final height(s) and slope(s) of the
                rays after tracing.
        """
        y = self._process_input(y)
        u = self._process_input(u)
        z = self._process_input(z)

        R = self.optic.surface_group.radii
        n = self.optic.n(wavelength)
        pos = be.ravel(self.optic.surface_group.positions)
        surfs = self.optic.surface_group.surfaces

        if reverse:
            R = -be.flip(R)
            n = be.roll(n, shift=1)
            n = be.flip(n)
            pos = pos[-1] - be.flip(pos)
            surfs = surfs[::-1]

        power = be.diff(n, prepend=be.array([1])) / R

        heights = []
        slopes = []

        for k in range(skip, len(R)):
            if isinstance(surfs[k], ObjectSurface):
                heights.append(be.copy(y))
                slopes.append(be.copy(u))
                continue

            # propagate to surface
            t = pos[k] - z
            z = pos[k]
            y = y + t * u

            # reflect or refract
            if surfs[k].is_reflective:
                u = -u - 2 * y / R[k]
            else:
                u = 1 / n[k] * (n[k - 1] * u - y * power[k])

            heights.append(be.copy(y))
            slopes.append(be.copy(u))

        heights = be.array(heights).reshape(-1, 1)
        slopes = be.array(slopes).reshape(-1, 1)

        return heights, slopes

    def _get_object_position(self, Hy, y1, EPL):
        """Calculate the position of the object in the paraxial optical system.

        Args:
            Hy (float): The normalized field height.
            y1 (ndarray): The initial y-coordinate of the ray.
            EPL (float): The effective focal length of the lens.

        Returns:
            tuple: A tuple containing the y and z coordinates of the object
                position.

        Raises:
            ValueError: If the field type is "object_height" and the object is
                at infinity.

        """
        obj = self.optic.object_surface
        field_y = self.optic.fields.max_field * Hy

        if obj.is_infinite:
            if self.optic.field_type == "object_height":
                raise ValueError(
                    'Field type cannot be "object_height" for an object at infinity.',
                )

            y = -be.tan(be.radians(field_y)) * EPL
            z = self.optic.surface_group.positions[1]

            y0 = y1 + y
            z0 = be.ones_like(y1) * z
        elif self.optic.field_type == "object_height":
            y = -field_y
            z = obj.geometry.cs.z

            y0 = be.ones_like(y1) * y
            z0 = be.ones_like(y1) * z

        elif self.optic.field_type == "angle":
            y = -be.tan(be.radians(field_y))
            z = self.optic.surface_group.positions[0]

            y0 = y1 + y
            z0 = be.ones_like(y1) * z

        return y0, z0

    def _process_input(self, x):
        """
        Process input to ensure it is a numpy array.

        Args:
            x (float or array-like): The input to process.

        Returns:
            np.ndarray: The processed input.
        """
        if isinstance(x, (int, float)):
            return be.array([x])
        else:
            return be.array(x)
