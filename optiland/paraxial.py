"""Paraxial Module

This module provides various functionalities for the computation of paraxial
properties of lens systems.

Note that object-space coordinates are defined relative to the first surface
(at index 1), while image-space coordinates are defined relative to the image surface.
This is relevant for the focal points (F1 & F2), principal planes (P1 & P2),
anti-principal planes (P1anti & P2anti), nodal planes (N1 & N2), and anti-nodal
planes (N1anti & N2anti). In the Optiland convention, the 1 denotes object space and
the 2 denotes image space. For example, P1 is the object space principle plane and F2
is the back focal point.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.raytrace.paraxial_ray_tracer import ParaxialRayTracer


class Paraxial:
    """A class representing a paraxial optical system.

    This class provides methods to calculate various properties of the optical
    system, such as focal lengths, entrance pupil location, exit pupil
    location, entrance pupil diameter, exit pupil diameter, image-space
    F-number, magnification, and more.

    Args:
        optic (Optic): The optical system to analyze.

    Attributes:
        optic (Optic): The optical system being analyzed.
        surfaces (SurfaceGroup): The surface group of the optical system.

    """

    def __init__(self, optic):
        self.optic = optic
        self._ray_tracer = ParaxialRayTracer(self.optic)

    @property
    def surfaces(self):
        """SurfaceGroup: the surface group of the optical system."""
        return self.optic.surface_group

    def f1(self):
        """Calculate the front focal length (f1).

        Returns:
            float: Front focal length.

        """
        z_start = -1
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(1.0, 0.0, z_start, wavelength, reverse=True)
        f1 = y[0] / u[-1]
        return f1[0]

    def f2(self):
        """Calculate the back focal length (f2), also known as effective focal length.

        Returns:
            float: Back focal length.

        """
        # start tracing 1 lens unit before first surface
        z_start = self.surfaces.positions[1] - 1
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(1.0, 0.0, z_start, wavelength)
        f2 = -y[0] / u[-1]
        return be.abs(f2[0])

    def F1(self):
        """Calculate the front focal point (F1) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            float: Front focal point location.

        """
        # start tracing 1 lens unit before first surface
        z_start = -1
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(1.0, 0.0, z_start, wavelength, reverse=True)
        F1 = y[-1] / u[-1]
        return F1[0]

    def F2(self):
        """Calculate the back focal point (F2) location.

        Note that this is defined relative to the image surface location.

        Returns:
            float: Back focal point location.

        """
        # start tracing 1 lens unit before first surface
        z_start = self.surfaces.positions[1] - 1
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(1.0, 0.0, z_start, wavelength)
        F2 = -y[-1] / u[-1]
        return F2[0]

    def P1(self):
        """Calculate the front principal plane (P1) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            float: Front principal plane location.

        """
        return self.F1() - self.f1()

    def P2(self):
        """Calculate the back principal plane (P2) location.

        Note that this is defined relative to the image surface location.

        Returns:
            float: Back principal plane location.

        """
        return self.F2() - self.f2()

    def P1anti(self):
        """Calculate the front anti-principal plane (P1anti) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            float: Front anti-principal plane location.
        """
        return self.F1() + self.f1()

    def P2anti(self):
        """Calculate the back anti-principal plane (P2anti) location.

        Note that this is defined relative to the image surface location.

        Returns:
            float: Back anti-principal plane location.
        """
        return self.F2() + self.f2()

    def N1(self):
        """Calculate the front nodal plane (N1) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            float: Front nodal plane location.

        """
        return self.F1() + self.f2()

    def N2(self):
        """Calculate the back nodal plane (N2) location.

        Note that this is defined relative to the image surface location.

        Returns:
            float: Back nodal plane location.

        """
        return self.F2() + self.f1()

    def N1anti(self):
        """Calculate the front anti-nodal plane (N1anti) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            float: Front anti-nodal plane location.

        """
        return self.F1() - self.f2()

    def N2anti(self):
        """Calculate the back anti-nodal plane (N2anti) location.

        Note that this is defined relative to the image surface location.

        Returns:
            float: Back anti-nodal plane location.

        """
        return self.F2() - self.f1()

    def EPL(self):
        """Calculate the entrance pupil location (EPL) in global coordinates.

        Returns:
            float: Entrance pupil position relative to the first surface
                (which lies at z=0 by definition in its local coordinate system).

        """
        stop_index = self.surfaces.stop_index
        if stop_index == 1:
            return self.surfaces.positions[1, 0]

        y0 = 0
        u0 = 0.1
        pos = self.surfaces.positions
        z0 = pos[-1] - pos[stop_index]
        wavelength = self.optic.primary_wavelength

        # trace from center of stop on axis
        skip = self.surfaces.num_surfaces - stop_index
        y, u = self._trace_generic(y0, u0, z0[0], wavelength, reverse=True, skip=skip)

        loc_relative = y[-1] / u[-1]
        return loc_relative[0]

    def EPD(self):
        """Calculate the entrance pupil diameter (EPD).

        Returns:
            float: Entrance pupil diameter.

        """
        ap_type = self.optic.aperture.ap_type
        ap_value = self.optic.aperture.value

        if ap_type == "EPD":
            return ap_value

        elif ap_type == "imageFNO":
            return self.f2() / ap_value

        elif ap_type == "objectNA":
            obj_z = self.optic.object_surface.geometry.cs.z
            wavelength = self.optic.primary_wavelength
            n0 = self.optic.object_surface.material_post.n(wavelength)
            u0 = be.arcsin(ap_value / n0)
            z = self.EPL() - obj_z
            return 2 * z * be.tan(u0)

        elif ap_type == "float_by_stop_size":
            stop_index = self.surfaces.stop_index
            wavelength = self.optic.primary_wavelength
            if self.optic.object_surface.is_infinite:
                y, _ = self._trace_generic(1.0, 0.0, -1, wavelength)
                return ap_value / y[stop_index]
            else:
                obj_z = self.optic.object_surface.geometry.cs.z
                EPL = self.EPL()
                y, _ = self._trace_generic(0.0, 0.1, obj_z, wavelength)
                u0 = 0.1 * ap_value / y[stop_index]
                return u0 * (EPL - obj_z)

    def XPL(self):
        """Calculate the exit pupil location (XPL).

        Returns:
            float: Exit pupil location relative to the image surface.

        """
        stop_index = self.surfaces.stop_index
        z_start = self.surfaces.positions[stop_index]
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(0.0, 0.1, z_start, wavelength, skip=stop_index + 1)
        loc_relative = -y[-1] / u[-1]
        return loc_relative[0]

    def XPD(self):
        """Calculate the exit pupil diameter (XPD).

        Returns:
            float: Exit pupil diameter.

        """
        # find marginal ray height at image surface
        ya, ua = self.marginal_ray()
        yi = ya[-1]
        ui = ua[-1]

        # find distance from image surface to exit pupil location
        xpl = self.XPL()

        # propagate marginal ray to this location
        yxp = yi + ui * xpl
        return 2 * yxp[0]

    def FNO(self):
        """Calculate the image-space F-number (FNO).

        Returns:
            float: Image-space F-number.

        """
        ap_type = self.optic.aperture.ap_type
        if ap_type == "imageFNO":
            return self.optic.aperture.value
        return self.f2() / self.EPD()

    def magnification(self):
        """Calculate the transverse magnification.

        Returns:
            float: The system's transverse magnification.

        """
        _, ua = self.marginal_ray()
        n = self.optic.n()
        mag = n[0] * ua[0] / (n[-1] * ua[-1])
        return mag[0]

    def invariant(self):
        """Calculate the Lagrange invariant.

        Returns:
            float: The Lagrange invariant of the system.

        """
        ya, ua = self.marginal_ray()
        yb, ub = self.chief_ray()
        n = self.optic.n()
        inv = yb[1] * n[1] * ua[1] - ya[1] * n[1] * ub[1]
        return inv[0]

    def marginal_ray(self):
        """Calculates the marginal ray heights and angles at each surface.

        The marginal ray originates from the center of the object and passes
        through the edge of the aperture stop.

        Returns:
            tuple[be.ndarray, be.ndarray]: A tuple containing two arrays:
                - y_marginal: Heights of the marginal ray at each surface.
                - u_marginal: Slopes of the marginal ray after each surface.

        """
        EPD = self.EPD()
        obj_z = self.surfaces.positions[1] - 10  # 10 mm before first surface
        if self.optic.object_surface.is_infinite:
            ya = EPD / 2
            ua = 0
        else:
            obj_z = self.optic.object_surface.geometry.cs.z
            z = self.EPL() - obj_z
            ya = 0
            ua = EPD / (2 * z)

        wavelength = self.optic.primary_wavelength
        return self._trace_generic(ya, ua, obj_z, wavelength)

    def chief_ray(self):
        """Calculates the chief ray heights and angles at each surface.

        The chief ray originates from the edge of the field of view and passes
        through the center of the aperture stop.

        Returns:
            tuple[be.ndarray, be.ndarray]: A tuple containing two arrays:
                - y_chief: Heights of the chief ray at each surface.
                - u_chief: Slopes of the chief ray after each surface.

        """
        stop_index = self.optic.surface_group.stop_index
        y0 = 0
        u0 = 0.1
        pos = self.optic.surface_group.positions
        z0 = pos[-1] - pos[stop_index]
        wavelength = self.optic.primary_wavelength
        num_surf = self.surfaces.num_surfaces
        skip = num_surf - stop_index

        # trace from center of stop on axis
        y, u = self._trace_generic(y0, u0, z0, wavelength, reverse=True, skip=skip)

        max_field = self.optic.fields.max_y_field

        if self.optic.field_type == "object_height":
            u1 = 0.1 * max_field / y[-1]
        elif self.optic.field_type == "angle":
            u1 = 0.1 * be.tan(be.deg2rad(max_field)) / u[-1]

        yn, un = self._trace_generic(y0, u1, z0, wavelength, reverse=True, skip=skip)

        # trace in forward direction
        z0 = self.optic.surface_group.positions[1]

        return self._trace_generic(-yn[-1, 0], un[-1, 0], z0[0], wavelength)

    def trace(self, Hy, Py, wavelength):
        """Trace paraxial ray through the optical system based on specified field
        and pupil coordinates.

        Args:
            Hy (float): Normalized field coordinate (typically in y).
            Py (float): Normalized pupil coordinate (typically in y).
            wavelength (float): Wavelength of the light in micrometers.

        Returns:
            tuple[be.ndarray, be.ndarray]: A tuple containing two arrays:
                - y_ray: Heights of the traced ray at each surface.
                - u_ray: Slopes of the traced ray after each surface.
        """
        return self._ray_tracer.trace(Hy, Py, wavelength)

    def _trace_generic(self, y, u, z, wavelength, reverse=False, skip=0):
        """Trace generically-defined paraxial rays through the optical system.

        Args:
            y (float or be.ndarray): The initial height(s) of the rays.
            u (float or be.ndarray): The initial slope(s) of the rays.
            z (float or be.ndarray): The initial axial position(s) of the rays,
                relative to the first surface if tracing forward, or relative
                to the last surface if tracing in reverse (before internal reversal).
            wavelength (float): The wavelength of the rays in micrometers.
            reverse (bool, optional): If True, trace the rays in reverse
                direction (from image to object space). Defaults to False.
            skip (int, optional): The number of surfaces to skip from the
                beginning of the trace (or end if reverse). Defaults to 0.

        Returns:
            tuple[be.ndarray, be.ndarray]: A tuple containing the height(s)
                and slope(s) of the rays at each surface interface after tracing.

        """
        return self._ray_tracer.trace_generic(y, u, z, wavelength, reverse, skip)
