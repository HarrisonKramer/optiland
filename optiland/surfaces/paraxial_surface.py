"""Paraxial Surface

This module contains the class ParaxialSurface, which is a subclass of the
Surface class. It is used to represent a paraxial surface, or a thin lens,
which is defined simply by its effective focal length in air. This class is
used to model the behavior of a lens in the paraxial approximation and can be
used for first-order layout of optical systems.

Kramer Harrison, 2024
"""

import numpy as np

from optiland.coatings import BaseCoating
from optiland.geometries import BaseGeometry
from optiland.materials import BaseMaterial
from optiland.physical_apertures import BaseAperture
from optiland.rays.polarized_rays import PolarizedRays
from optiland.scatter import BaseBSDF
from optiland.surfaces.standard_surface import Surface


class ParaxialSurface(Surface):
    """Paraxial Surface

    This class is used to represent a paraxial surface, which is planar, yet
    imparts a phase shift to the incident light, causing refraction. This class
    is used to model the behavior of a lens in the paraxial approximation and
    can be used for first-order layout of optical systems.
    """

    def __init__(
        self,
        focal_length,
        geometry,
        material_pre,
        material_post,
        is_stop=False,
        aperture=None,
        coating=None,
        bsdf=None,
        is_reflective=False,
        surface_type="paraxial",
    ):
        self.f = focal_length
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

    def _interact(self, rays):
        """Interacts the rays with the surface by either reflecting or refracting

        Note that phase is added assuming a thin lens as a phase
        transformation. A cosine correction is applied for rays propagating
        off-axis. This correction is equivalent to the ray z direction cosine.

        Args:
            rays: The rays.

        Returns:
            RealRays: The refracted rays.

        """
        # add optical path length - workaround for now
        # TODO: develop more robust method
        rays.opd -= (rays.x**2 + rays.y**2) / (2 * self.f * rays.N)

        n1 = self.material_pre.n(rays.w)

        n2 = -n1 if self.is_reflective else self.material_post.n(rays.w)

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
            rays = self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=0,
                ny=0,
                nz=1,
            )
        else:
            # update polarization matrices, if PolarizedRays
            rays.update()

        # paraxial approximation -> direction is not necessarily unit vector
        rays.L = L
        rays.M = M
        rays.N = np.ones_like(L)
        rays.is_normalized = False

        return rays

    def _trace_paraxial(self, rays):
        """Traces paraxial rays through the surface.

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

    def to_dict(self):
        """Returns a dictionary representation of the surface."""
        return {
            "type": self.__class__.__name__,
            "focal_length": self.f,
            "geometry": self.geometry.to_dict(),
            "material_pre": self.material_pre.to_dict(),
            "material_post": self.material_post.to_dict(),
            "is_stop": self.is_stop,
            "aperture": self.aperture.to_dict() if self.aperture else None,
            "coating": self.coating.to_dict() if self.coating else None,
            "bsdf": self.bsdf.to_dict() if self.bsdf else None,
            "is_reflective": self.is_reflective,
        }

    @classmethod
    def _from_dict(cls, data):
        """Protected deserialization logic for direct initialization.

        Args:
            data (dict): The dictionary representation of the surface.

        Returns:
            Surface: The surface.

        """
        focal_length = data["focal_length"]
        geometry = BaseGeometry.from_dict(data["geometry"])
        material_pre = BaseMaterial.from_dict(data["material_pre"])
        material_post = BaseMaterial.from_dict(data["material_post"])
        aperture = (
            BaseAperture.from_dict(data["aperture"]) if data["aperture"] else None
        )
        coating = BaseCoating.from_dict(data["coating"]) if data["coating"] else None
        bsdf = BaseBSDF.from_dict(data["bsdf"]) if data["bsdf"] else None

        return ParaxialSurface(
            focal_length,
            geometry,
            material_pre,
            material_post,
            data["is_stop"],
            aperture,
            coating,
            bsdf,
            data["is_reflective"],
        )
