"""Interaction model for a thin lens

This module implements the ThinLensInteractionModel class, which handles
ray interactions with a thin lens surface.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.interactions.base import BaseInteractionModel
from optiland.rays.polarized_rays import PolarizedRays

if TYPE_CHECKING:
    # pragma: no cover
    from optiland.coatings import BaseCoating
    from optiland.geometries.base import BaseGeometry
    from optiland.materials.base import BaseMaterial
    from optiland.scatter import BaseBSDF


class ThinLensInteractionModel(BaseInteractionModel):
    """Interaction model for a thin lens."""

    def __init__(
        self,
        focal_length: float,
        geometry: BaseGeometry,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
        is_reflective: bool,
        coating: BaseCoating | None = None,
        bsdf: BaseBSDF | None = None,
    ):
        super().__init__(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            is_reflective=is_reflective,
            coating=coating,
            bsdf=bsdf,
        )
        self.f = be.array(focal_length)

    def to_dict(self):
        """Returns a dictionary representation of the thin lens model."""
        data = super().to_dict()
        data["focal_length"] = self.f.item()
        return data

    def flip(self):
        """Flip the interaction model."""
        self.material_pre, self.material_post = self.material_post, self.material_pre
        self.f = -self.f

    def interact_real_rays(self, rays):
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
        rays.opd = rays.opd - (rays.x**2 + rays.y**2) / (2 * self.f)

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
        rays.N = be.ones_like(L)
        rays.is_normalized = False

        return rays

    def interact_paraxial_rays(self, rays):
        """Traces paraxial rays through the surface.

        Args:
            ParaxialRays: The paraxial rays to be traced.

        """
        n1 = self.material_pre.n(rays.w)
        if self.is_reflective:
            # reflect (derived from paraxial equations when n'=-n)
            rays.u = rays.y / (self.f * n1) - rays.u

        else:
            # surface power
            n2 = self.material_post.n(rays.w)

            # refract
            rays.u = 1 / n2 * (n1 * rays.u - rays.y / self.f)

        return rays
