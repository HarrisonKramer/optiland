from copy import deepcopy
import numpy as np
from optiland.rays import BaseRays, RealRays, ParaxialRays
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane, StandardGeometry
from optiland import materials


class Surface:

    def __init__(self, geometry, material_pre, material_post,
                 is_stop=False, aperture=None):
        self.geometry = geometry
        self.material_pre = material_pre
        self.material_post = material_post
        self.is_stop = is_stop
        self.aperture = aperture
        self.semi_aperture = None

        self.reset()

    def trace(self, rays: BaseRays):
        if isinstance(rays, ParaxialRays):
            return self._trace_paraxial(rays)
        elif isinstance(rays, RealRays):
            return self._trace_real(rays)

    def set_semi_aperture(self, r_max: float):
        self.semi_aperture = r_max

    def reset(self):
        self.y = np.empty(0)
        self.u = np.empty(0)
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.z = np.empty(0)

        self.L = np.empty(0)
        self.M = np.empty(0)
        self.N = np.empty(0)

        self.energy = np.empty(0)
        self.aoi = np.empty(0)
        self.opd = np.empty(0)

    def _compute_aoi(self, rays, nx, ny, nz):
        dot = np.abs(nx * rays.L + ny * rays.M + nz * rays.N)
        dot = np.clip(dot, -1, 1)  # required due to numerical precision
        return np.arccos(dot) / 2

    def _record(self, rays, aoi=None):
        if isinstance(rays, ParaxialRays):
            self.y = np.copy(np.atleast_1d(rays.y))
            self.u = np.copy(np.atleast_1d(rays.u))
        elif isinstance(rays, RealRays):
            self.x = np.copy(np.atleast_1d(rays.x))
            self.y = np.copy(np.atleast_1d(rays.y))
            self.z = np.copy(np.atleast_1d(rays.z))

            self.L = np.copy(np.atleast_1d(rays.L))
            self.M = np.copy(np.atleast_1d(rays.M))
            self.N = np.copy(np.atleast_1d(rays.N))

            self.energy = np.copy(np.atleast_1d(rays.e))
            self.aoi = np.copy(np.atleast_1d(aoi))
            self.opd = np.copy(np.atleast_1d(rays.opd))

    def _interact(self, rays, nx, ny, nz):
        ix = rays.L
        iy = rays.M
        iz = rays.N

        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        u = n1 / n2
        ni = nx*ix + ny*iy + nz*iz
        root = np.sqrt(1 - u**2 * (1 - ni**2))
        tx = u * ix + nx * root - u * nx * ni
        ty = u * iy + ny * root - u * ny * ni
        tz = u * iz + nz * root - u * nz * ni

        rays.L = tx
        rays.M = ty
        rays.N = tz

        return rays

    def _trace_paraxial(self, rays: ParaxialRays):
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        # propagate to this surface
        t = -rays.z
        rays.propagate(t)

        # surface power
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)
        power = (n2 - n1) / self.geometry.radius

        # refract
        rays.u = 1 / n2 * (n1 * rays.u - rays.y * power)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        self._record(rays)

    def _trace_real(self, rays: RealRays):
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        # find distance from rays to the surface
        t = self.geometry.distance(rays)

        # propagate the rays a distance t
        rays.propagate(t)

        # update OPD
        rays.opd += t * self.material_pre.n(rays.w)

        # if there is a limiting aperture, clip rays outside of it
        if self.aperture:
            self.aperture.clip(rays)

        # find surface normals
        nx, ny, nz = self.geometry.surface_normal(rays)

        # find AOIs
        aoi = self._compute_aoi(rays, nx, ny, nz)

        # Interact with surface (refract or reflect)
        rays = self._interact(rays, nx, ny, nz)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        # record ray information
        self._record(rays, aoi)

        return rays


class ObjectSurface(Surface):

    def __init__(self, geometry, material_post):
        super().__init__(
            geometry=geometry,
            material_pre=material_post,
            material_post=material_post,
            is_stop=False,
            aperture=None
        )

    @property
    def is_infinite(self):
        return np.isinf(self.geometry.cs.z)

    def set_aperture(self):
        pass

    def trace(self, rays):
        # reset recorded information
        self.reset()

        # record ray information
        self._record(rays)

        return rays

    def _trace_paraxial(self, rays: ParaxialRays):
        pass

    def _trace_real(self, rays: ParaxialRays):
        pass

    def _interact(self, rays, nx, ny, nz):
        return rays


class ImageSurface(Surface):

    def __init__(self, geometry, material_pre, aperture=None):
        super().__init__(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_pre,
            is_stop=False,
            aperture=aperture
        )

    def _trace_paraxial(self, rays: ParaxialRays):
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        # propagate to this surface
        t = -rays.z
        rays.propagate(t)

        self._record(rays)

    def _interact(self, rays, nx, ny, nz):
        return rays


class SurfaceGroup:

    def __init__(self, surfaces=[]):
        self.surfaces = surfaces

        self._last_thickness = 0

    @property
    def x(self):
        return np.array([surf.x for surf in self.surfaces if surf.x.size > 0])

    @property
    def y(self):
        return np.array([surf.y for surf in self.surfaces if surf.y.size > 0])

    @property
    def z(self):
        return np.array([surf.z for surf in self.surfaces if surf.z.size > 0])

    @property
    def L(self):
        return np.array([surf.L for surf in self.surfaces if surf.L.size > 0])

    @property
    def M(self):
        return np.array([surf.M for surf in self.surfaces if surf.M.size > 0])

    @property
    def N(self):
        return np.array([surf.N for surf in self.surfaces if surf.N.size > 0])

    @property
    def opd(self):
        return np.array([surf.opd for surf in self.surfaces if surf.opd.size > 0])

    @property
    def u(self):
        return np.array([surf.u for surf in self.surfaces if surf.u.size > 0])

    @property
    def energy(self):
        return np.array([surf.energy for surf in self.surfaces if surf.energy.size > 0])

    @property
    def positions(self):
        return np.array([surf.geometry.cs.position_in_gcs[2] for surf in self.surfaces])

    @property
    def radii(self):
        return np.array([surf.geometry.radius for surf in self.surfaces])

    @property
    def stop_index(self):
        for index, surface in enumerate(self.surfaces):
            if surface.is_stop:
                return index

    @property
    def num_surfaces(self):
        return len(self.surfaces)

    def get_thickness(self, surface_number):
        t = self.positions
        return t[surface_number+1] - t[surface_number+1]

    def trace(self, rays, skip=0):
        self.reset()
        for surface in self.surfaces[skip:]:
            surface.trace(rays)

    def add_surface(self, new_surface=None, index=None, thickness=0, radius=np.inf,
                    material='air', conic=0, is_stop=False, dx=0, dy=0, rx=0, ry=0):
        if new_surface is None:
            if index is None:
                raise ValueError('Must define index when defining surface.')

            new_surface = self._configure_surface(index, thickness, radius, material, conic,
                                                  is_stop, dx, dy, rx, ry)

        if new_surface.is_stop:
            for surface in self.surfaces:
                surface.is_stop = False

        self.surfaces.insert(index, new_surface)

    def remove_surface(self, index):
        if index == 0:
            raise ValueError('Cannot remove object surface.')
        elif index == len(self.surfaces)-1:
            raise ValueError('Cannot remove image surface.')
        del self.surfaces[index]

    def reset(self):
        for surface in self.surfaces:
            surface.reset()

    def inverted(self):
        """Generate inverted surface group"""
        surfs_inverted = deepcopy(self.surfaces[::-1])
        z_shift = self.surfaces[-1].geometry.cs.z
        for surf in surfs_inverted:
            # scale radii by -1
            surf.geometry.radius *= -1

            # invert z position
            surf.geometry.cs.z = z_shift - surf.geometry.cs.z

            # swap initial and final materials
            temp = surf.material_pre
            surf.material_pre = surf.material_post
            surf.material_post = temp

        return SurfaceGroup(surfs_inverted)

    def _configure_cs(self, index, thickness, dx, dy, rx, ry):
        if index == 0:  # object surface
            z = -thickness
        elif index == 1:
            z = 0  # first surface, always at zero
        else:
            z = self.positions[index-1] + self._last_thickness

        self._last_thickness = thickness

        return CoordinateSystem(x=dx, y=dy, z=z, rx=rx, ry=ry)

    def _configure_geometry(self, cs, radius, conic):
        if np.isinf(radius):
            geometry = Plane(cs)
        else:
            geometry = StandardGeometry(cs, radius, conic)

        return geometry

    def _configure_material(self, index, material):
        if isinstance(material, materials.BaseMaterial):
            material_post = material
        elif isinstance(material, tuple):
            material_post = materials.Material(name=material[0], manufacturer=material[1])
        elif isinstance(material, str):
            if material == 'mirror':
                material_post = materials.Mirror()
            elif material == 'air':
                material_post = materials.IdealMaterial(n=1.0, k=0.0)
            else:
                material_post = materials.Material(material)

        if index == 0:
            material_pre = None
        else:
            material_pre = self.surfaces[index-1].material_post

        return material_pre, material_post

    def _configure_surface(self, index, thickness=0, radius=np.inf, material='air', conic=0,
                           is_stop=False, dx=0, dy=0, rx=0, ry=0):
        if index > self.num_surfaces:
            raise ValueError('Surface index cannot be greater than number of surfaces.')

        cs = self._configure_cs(index, thickness, dx, dy, rx, ry)
        geometry = self._configure_geometry(cs, radius, conic)
        material_pre, material_post = self._configure_material(index, material)

        if index == 0:
            return ObjectSurface(geometry, material_post)
        elif index == self.num_surfaces-1:
            return ImageSurface(geometry, material_pre)
        else:
            return Surface(geometry, material_pre, material_post, is_stop)
