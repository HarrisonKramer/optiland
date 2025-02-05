"""Paraxial Surface

This module contains the class ParaxialSurface, which is a subclass of the
Surface class. It is used to represent a paraxial surface, which is planar,
yet imparts a phase shift to the incident light, causing refraction. This
class is used to model the behavior of a lens in the paraxial approximation
and can be used for first-order layout of optical systems.

Kramer Harrison, 2025
"""
import numpy as np
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.surfaces.standard_surface import Surface


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
                 ):
        self.f = focal_length
        cs = CoordinateSystem()
        geometry = Plane(cs)
        surface_type = 'paraxial'
        super().__init__(geometry, material_pre, material_post, is_stop,
                         aperture, coating, bsdf, is_reflective, surface_type)

    def _interact(self, rays):
        """
        Interacts the rays with the surface by either reflecting or refracting

        Args:
            rays: The rays.

        Returns:
            RealRays: The refracted rays.
        """
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        if self.is_reflective:
            n2 = -n2

        ux1 = rays.L / rays.N
        uy1 = rays.M / rays.N

        ux2 = 1 / n2 * (n1 * ux1 - rays.x / self.f)
        uy2 = 1 / n2 * (n1 * uy1 - rays.y / self.f)

        # New direction cosines (not normalized in paraxial approximation)
        L = ux2
        M = uy2
        N = 1

        # TODO: work out how to maintain non-normalized direction cosines,
        # as required for paraxial approximation (or is it?)

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

        return rays, L, M, N

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

        if self.is_reflective:
            # reflect (derived from paraxial equations when n'=-n)
            rays.u = -rays.u - 2 * rays.y / self.geometry.radius

        else:
            # surface power
            n1 = self.material_pre.n(rays.w)
            n2 = self.material_post.n(rays.w)
            power = (n2 - n1) / self.geometry.radius

            # refract
            rays.u = 1 / n2 * (n1 * rays.u - rays.y * power)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        self._record(rays)

        return rays

    def _trace_real(self, rays):
        """
        Traces real rays through the surface.

        Args:
            rays (RealRays): The real rays to be traced.

        Returns:
            RealRays: The traced real rays.
        """
        # reset recorded information
        self.reset()

        # transform coordinate system
        self.geometry.localize(rays)

        # find distance from rays to the surface
        t = self.geometry.distance(rays)

        # propagate the rays a distance t through material
        rays.propagate(t, self.material_pre)

        # update OPD
        rays.opd += np.abs(t * self.material_pre.n(rays.w))

        # if there is a limiting aperture, clip rays outside of it
        if self.aperture:
            self.aperture.clip(rays)

        # interact with surface
        rays = self._interact(rays)

        # inverse transform coordinate system
        self.geometry.globalize(rays)

        # record ray information
        self._record(rays)

        return rays
