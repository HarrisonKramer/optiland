"""Standard Surface

This module defines the `Surface` class, which represents a surface in an
optical system. Surfaces are characterized by their geometry, materials before
and after the surface, and optional properties such as being an aperture stop,
having a physical aperture, and a coating. The module facilitates the tracing
of rays through these surfaces, accounting for refraction, reflection, and
absorption based on the surface properties and materials involved.

Kramer Harrison, 2023
"""

from __future__ import annotations

import optiland.backend as be
from optiland.rays import BaseRays, ParaxialRays, RealRays
from optiland.surfaces.standard_surface import Surface


class GratingSurface(Surface):
    """Represents a grating surface in an optical system.

    Args:
        geometry (BaseGeometry): The geometry of the surface.
        material_pre (BaseMaterial): The material before the surface.
        material_post (BaseMaterial): The material after the surface.
        is_stop (bool, optional): Indicates if the surface is the aperture
            stop. Defaults to False.
        aperture (BaseAperture, int, float, optional): The physical aperture of the
            surface. Defaults to None. If a scalar is provided, it specifies the
            diameter of the lens.
        coating (BaseCoating, optional): The coating applied to the surface.
            Defaults to None.
        comment (str, optional): A comment for the surface. Defaults to ''.

    """

    _registry = {}  # registry for all surfaces

    def __init__(
        self,
        geometry,
        material_pre,
        material_post,
        is_stop=False,
        aperture=None,
        coating=None,
        bsdf=None,
        is_reflective=False,
        surface_type="grating",
    ):
        super().__init__(
            geometry,
            material_pre,
            material_post,
            is_stop,
            aperture,
            coating,
            bsdf,
            is_reflective,
            surface_type,
        )

    def trace(self, rays: BaseRays):
        """Traces the given rays through the surface.

        Args:
            rays (BaseRays): The rays to be traced.

        Returns:
            BaseRays: The traced rays.

        """
        if isinstance(rays, ParaxialRays):
            return self._trace_paraxial(rays)
        if isinstance(rays, RealRays):
            return self._trace_real(rays)

    def _interact(self, rays):
        """Interacts the rays with the surface by either reflecting or refracting

        Args:
            rays: The rays.

        Returns:
            RealRays: The diffracted rays.

        """
        # find surface normals
        nx, ny, nz = self.geometry.surface_normal(rays)

        # Interact with surface (refract or reflect)
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # find grating vector
        fx, fy, fz = self.geometry.grating_vector(rays)

        # grating period
        pp = self.geometry.grating_period

        # correct grating period considering projection effect on the surface
        pp = pp / be.sqrt(fx**2 + fy**2)

        # grating order
        m = self.geometry.grating_order

        rays.gratingdiffract(nx, ny, nz, fx, fy, fz, m, pp, n1, n2, self.is_reflective)

        # if there is a surface scatter model, modify ray properties
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx, ny, nz)

        # if there is a coating, modify ray properties
        if self.coating:
            rays = self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=nx,
                ny=ny,
                nz=nz,
            )
        else:
            # update polarization matrices, if PolarizedRays
            rays.update()

        return rays

    def _trace_paraxial(self, rays: ParaxialRays):
        """Traces paraxial rays through the surface.

        Note: To be modified for grating

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

    def _trace_real(self, rays: RealRays):
        """Traces real rays through the surface.

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
        rays.opd = rays.opd + be.abs(t * self.material_pre.n(rays.w))

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
