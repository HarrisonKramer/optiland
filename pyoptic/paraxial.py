from pyoptic.rays import ParaxialRays
import numpy as np


class Paraxial:

    def __init__(self, optic):
        self.optic = optic
        self.surfaces = self.optic.surface_group

    def f1(self):
        surfaces = self.surfaces.inverted()
        z_start = surfaces.positions[0] - 1  # start tracing 1 lens unit before first surface
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(1.0, 0.0, z_start, wavelength, reverse=True)
        f1 = y[0] / u[-1]
        return f1[0]

    def f2(self):
        z_start = self.surfaces.positions[1] - 1  # start tracing 1 lens unit before first surface
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(1.0, 0.0, z_start, wavelength)
        f2 = -y[0] / u[-1]
        return f2[0]

    def F1(self):
        surfaces = self.surfaces.inverted()
        z_start = surfaces.positions[0] - 1  # start tracing 1 lens unit before first surface
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(1.0, 0.0, z_start, wavelength, reverse=True)
        F1 = y[-1] / u[-1]
        return F1[0]

    def F2(self):
        z_start = self.surfaces.positions[1] - 1  # start tracing 1 lens unit before first surface
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(1.0, 0.0, z_start, wavelength)
        F2 = -y[-1] / u[-1]
        return F2[0]

    def P1(self):
        return self.F1() - self.f1()

    def P2(self):
        return self.F2() - self.f2()

    def N1(self):
        return self.P1() + self.f1() + self.f2()

    def N2(self):
        return self.P2() + self.f1() + self.f2()

    def EPL(self):
        """Entrance pupil location in global coordinates

        Trace ray from center of aperture stop into object space
        """
        stop_index = self.surfaces.stop_index
        if stop_index == 0:
            return self.surfaces.positions[1]

        surfaces = self.surfaces.inverted()
        stop_index = surfaces.stop_index

        y0 = 0
        u0 = 0.1
        z0 = surfaces.positions[stop_index]  # trace from center of stop on axis
        wavelength = self.optic.primary_wavelength

        y, u = self._trace_generic(y0, u0, z0, wavelength, reverse=True, skip=stop_index+1)

        loc_relative = y[-1] / u[-1]
        return loc_relative[0]

    def EPD(self):
        """Entrance pupil diameter"""
        ap_type = self.optic.aperture.ap_type
        ap_value = self.optic.aperture.value

        if ap_type == 'EPD':
            return ap_value

        elif ap_type == 'imageFNO':
            return self.f2() / ap_value

        elif ap_type == 'objectNA':
            obj_z = self.optic.object_surface.geometry.cs.z
            wavelength = self.optic.primary_wavelength
            n0 = self.optic.object_surface.material_post.n(wavelength)
            u0 = np.arcsin(ap_value / n0)
            z = self.EPL() - obj_z
            return 2 * z * np.tan(u0)

    def XPL(self):
        """Exit pupil location"""
        stop_index = self.surfaces.stop_index
        num_surfaces = len(self.surfaces.surfaces)
        if stop_index == num_surfaces-2:
            positions = self.optic.surface_group.positions
            return positions[-2] - positions[-1]

        z_start = self.surfaces.positions[stop_index]
        wavelength = self.optic.primary_wavelength
        y, u = self._trace_generic(0.0, 0.1, z_start, wavelength, skip=stop_index+1)

        loc_relative = -y[-1] / u[-1]
        return loc_relative[0]

    def XPD(self):
        """Exit pupil diameter"""
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
        """Image-space F-number"""
        ap_type = self.optic.aperture.ap_type
        if ap_type == 'imageFNO':
            return self.optic.aperture.value
        else:
            return self.f2() / self.EPD()

    def magnification(self):
        '''Magnification'''
        ya, ua = self.marginal_ray()
        n = self.optic.n()
        return n[0]*ua[0]/(n[-1]*ua[-1])

    def invariant(self):
        ya, ua = self.marginal_ray()
        yb, ub = self.chief_ray()
        n = self.optic.n()
        inv = yb[1] * n[1] * ua[1] - ya[1] * n[1] * ub[1]
        return inv[0]

    def marginal_ray(self):
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
        surfaces = self.surfaces.inverted()
        stop_index = surfaces.stop_index

        y0 = 0
        u0 = 0.1
        z0 = surfaces.positions[stop_index]  # trace from center of stop on axis
        wavelength = self.optic.primary_wavelength

        y, u = self._trace_generic(y0, u0, z0, wavelength, reverse=True, skip=stop_index+1)

        max_field = self.optic.fields.max_y_field

        if self.optic.field_type == 'object_height':
            u1 = 0.1 * max_field / y[-1]
        elif self.optic.field_type == 'angle':
            u1 = 0.1 * np.tan(np.deg2rad(max_field)) / u[-1]

        yn, un = self._trace_generic(y0, u1, z0, wavelength, reverse=True, skip=stop_index+1)

        # trace in forward direction
        z0 = self.surfaces.positions[1]

        return self._trace_generic(-yn[-1], un[-1], z0, wavelength)

    def _get_object_position(self, Hy, y1, EPL):
        obj = self.optic.object_surface
        field_y = self.optic.fields.max_field * Hy
        if obj.is_infinite:
            if self.optic.field_type == 'object_height':
                raise ValueError('Field type cannot be "object_height" for an object at infinity.')

            y = -np.tan(np.radians(field_y)) * EPL
            z = self.optic.surface_group.positions[1]

            y0 = y1 + y
            z0 = np.ones_like(y1) * z
        else:
            if self.optic.field_type == 'object_height':
                y = -field_y
                z = obj.geometry.cs.z

                y0 = np.ones_like(y1) * y
                z0 = np.ones_like(y1) * z

            elif self.optic.field_type == 'angle':
                y = -np.tan(np.radians(field_y))
                z = self.optic.surface_group.positions[0]

                y0 = y1 + y
                z0 = np.ones_like(y1) * z

        return y0, z0

    def trace(self, Hy, Py, wavelength):
        EPL = self.EPL()
        EPD = self.EPD()

        y1 = Py * EPD / 2

        y0, z0 = self._get_object_position(Hy, y1, EPL)
        u0 = (y1 - y0) / (EPL - z0)
        rays = ParaxialRays(y0, u0, z0, wavelength)

        self.optic.surface_group.trace(rays)

    def _trace_generic(self, y, u, z, wavelength, reverse=False, skip=0):
        if np.isscalar(y):
            y = np.array([y], dtype=float)
        else:
            y = np.array(y, dtype=float)

        if np.isscalar(u):
            u = np.array([u], dtype=float)
        else:
            u = np.array(u, dtype=float)

        if np.isscalar(z):
            z = np.array([z], dtype=float)
        else:
            z = np.array(z, dtype=float)

        if reverse:
            surfaces = self.surfaces.inverted()
        else:
            surfaces = self.surfaces

        rays = ParaxialRays(y, u, z, wavelength)
        surfaces.trace(rays, skip)

        return surfaces.y, surfaces.u
