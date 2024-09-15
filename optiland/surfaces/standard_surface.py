"""Optiland Surfaces Module

This module defines the `Surface` class, which represents a surface in an
optical system. Surfaces are characterized by their geometry, materials before
and after the surface, and optional properties such as being an aperture stop,
having a physical aperture, and a coating. The module facilitates the tracing
of rays through these surfaces, accounting for refraction, reflection, and
absorption based on the surface properties and materials involved.

Kramer Harrison, 2023
"""
import numpy as np
from optiland.rays import BaseRays, RealRays, ParaxialRays
from optiland.materials import BaseMaterial
from optiland.physical_apertures import BaseAperture
from optiland.scatter import BaseBSDF
from optiland.geometries import BaseGeometry
from optiland.coatings import BaseCoating, FresnelCoating


class Surface:
    """
    Represents a standard refractice surface in an optical system.

    Args:
        geometry (BaseGeometry): The geometry of the surface.
        material_pre (BaseMaterial): The material before the surface.
        material_post (BaseMaterial): The material after the surface.
        is_stop (bool, optional): Indicates if the surface is the aperture
            stop. Defaults to False.
        aperture (BaseAperture, optional): The physical aperture of the
            surface. Defaults to None.
        coating (BaseCoating, optional): The coating applied to the surface.
            Defaults to None.
    """

    def __init__(self,
                 geometry: BaseGeometry,
                 material_pre: BaseMaterial,
                 material_post: BaseMaterial,
                 is_stop: bool = False,
                 aperture: BaseAperture = None,
                 coating: BaseCoating = None,
                 bsdf: BaseBSDF = None,
                 is_reflective: bool = False):
        self.geometry = geometry
        self.material_pre = material_pre
        self.material_post = material_post
        self.is_stop = is_stop
        self.aperture = aperture
        self.semi_aperture = None
        self.coating = coating
        self.bsdf = bsdf
        self.is_reflective = is_reflective

        self.reset()

    def trace(self, rays: BaseRays):
        """
        Traces the given rays through the surface.

        Args:
            rays (BaseRays): The rays to be traced.

        Returns:
            BaseRays: The traced rays.
        """
        if isinstance(rays, ParaxialRays):
            return self._trace_paraxial(rays)
        elif isinstance(rays, RealRays):
            return self._trace_real(rays)

    def set_semi_aperture(self, r_max: float):
        """
        Sets the physical semi-aperture of the surface.

        Args:
            r_max (float): The maximum radius of the semi-aperture.
        """
        self.semi_aperture = r_max

    def reset(self):
        """
        Resets the recorded information of the surface.
        """
        self.y = np.empty(0)
        self.u = np.empty(0)
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.z = np.empty(0)

        self.L = np.empty(0)
        self.M = np.empty(0)
        self.N = np.empty(0)

        self.intensity = np.empty(0)
        self.aoi = np.empty(0)
        self.opd = np.empty(0)

    def set_fresnel_coating(self):
        """
        Sets the coating of the surface to a Fresnel coating.
        """
        self.coating = FresnelCoating(self.material_pre, self.material_post)

    def _record(self, rays):
        """
        Records the ray information.

        Args:
            rays: The rays.
        """
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

            self.intensity = np.copy(np.atleast_1d(rays.i))
            self.opd = np.copy(np.atleast_1d(rays.opd))

    def _interact(self, rays):
        """
        Interacts the rays with the surface by either reflecting or refracting

        Args:
            rays: The rays.

        Returns:
            RealRays: The refracted rays.
        """
        # find surface normals
        nx, ny, nz = self.geometry.surface_normal(rays)

        # Interact with surface (refract or reflect)
        if self.is_reflective:
            rays.reflect(nx, ny, nz)
        else:
            n1 = self.material_pre.n(rays.w)
            n2 = self.material_post.n(rays.w)
            rays.refract(nx, ny, nz, n1, n2)

        # if there is a surface scatter model, modify ray properties
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx, ny, nz)

        # if there is a coating, modify ray properties
        if self.coating:
            rays = self.coating.interact(rays, reflect=self.is_reflective,
                                         nx=nx, ny=ny, nz=nz)
        else:
            # update polarization matrices, if PolarizedRays
            rays.update()

        return rays

    def _trace_paraxial(self, rays: ParaxialRays):
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

    def _trace_real(self, rays: RealRays):
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