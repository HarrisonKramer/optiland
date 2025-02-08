"""Paraxial Surface

This module contains the class ParaxialSurface, which is a subclass of the
Surface class. It is used to represent a paraxial surface, or a thin lens,
which is defined simply by its effective focal length in air. This class is
used to model the behavior of a lens in the paraxial approximation and can be
used for first-order layout of optical systems.

Kramer Harrison, 2024
"""
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.surfaces.standard_surface import Surface
from optiland.rays.polarized_rays import PolarizedRays


class ParaxialSurface(Surface):
    """Paraxial Surface

    This class is used to represent a paraxial surface, which is planar, yet
    imparts a phase shift to the incident light, causing refraction. This class
    is used to model the behavior of a lens in the paraxial approximation and
    can be used for first-order layout of optical systems.
    """

    def __init__(self,
                 focal_length,
                 material_pre,
                 material_post,
                 is_stop=False,
                 aperture=None,
                 coating=None,
                 bsdf=None,
                 is_reflective=False,
                 surface_type='paraxial'):
        self.f = focal_length
        cs = CoordinateSystem()
        geometry = Plane(cs)
        super().__init__(geometry, material_pre, material_post, is_stop,
                         aperture, coating, bsdf, is_reflective, surface_type)

    def _interact(self, rays):
        """
        Interacts the rays with the surface by either reflecting or refracting

        Note that phase is added assuming a thin lens as a phase
        transformation. A cosine correction is applied for rays propagating
        off-axis. This correction is equivalent to the ray z direction cosine.

        Args:
            rays: The rays.

        Returns:
            RealRays: The refracted rays.
        """
        # add optical path length
        rays.opd += (rays.x**2 + rays.y**2) / (2 * self.f * rays.N)

        n1 = self.material_pre.n(rays.w)

        if self.is_reflective:
            n2 = -n1
        else:
            n2 = self.material_post.n(rays.w)

        ux1 = rays.L / rays.N
        uy1 = rays.M / rays.N

        ux2 = 1 / n2 * (n1 * ux1 - rays.x / self.f)
        uy2 = 1 / n2 * (n1 * uy1 - rays.y / self.f)

        L = ux2
        M = uy2

        # only normalize if required
        if self.bsdf or self.coating or isinstance(rays, PolarizedRays):
            rays.normalize()

        # if there is a surface scatter model, modify ray properties
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx=0, ny=0, nz=1)

        # if there is a coating, modify ray properties
        if self.coating:
            rays = self.coating.interact(rays, reflect=self.is_reflective,
                                         nx=0, ny=0, nz=1)
        else:
            # update polarization matrices, if PolarizedRays
            rays.update()

        # paraxial approximation -> direction is not necessarily unit vector
        rays.L = L
        rays.M = M
        rays.N = 1
        rays.is_normalized = False

        return rays

    def _trace_paraxial(self, rays):
        """
        Traces paraxial rays through the surface.

        Args:
            ParaxialRays: The paraxial rays to be traced.
        """
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        # propagate to this surface
        t = -rays.z
        rays.propagate(t)

        n1 = self.material_pre.n(rays.w)
        if self.is_reflective:
            # reflect (derived from paraxial equations when n'=-n)
            rays.u = rays.y / (self.f * n1) - rays.u

        else:
            # surface power
            n2 = self.material_post.n(rays.w)

            # refract
            rays.u = 1 / n2 * (n1 * rays.u - rays.y / self.f)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        self._record(rays)

        return rays
