"""Phase Interaction Model

This module defines the `PhaseInteractionModel` class, which represents
the interaction of rays with a surface with a phase profile.

The `PhaseInteractionModel` uses a `BasePhase` object to calculate the
effect of a phase function on the rays. It updates the ray's direction and
optical path difference based on the results of the phase calculation.

This class is designed to be used with the `Surface` class to model
optical elements with arbitrary phase profiles.

Hhsoj, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.interactions.base import BaseInteractionModel

if TYPE_CHECKING:
    from optiland.geometries import BaseGeometry
    from optiland.materials import BaseMaterial
    from optiland.phase.base import BasePhase
    from optiland.rays import RealRays, ParaxialRays


class PhaseInteractionModel(BaseInteractionModel):
    """Represents the interaction of rays with a diffractive surface.

    Args:
        geometry (BaseGeometry): The geometry of the surface.
        material_pre (BaseMaterial): The material before the surface.
        material_post (BaseMaterial): The material after the surface.
        phase_model (BasePhase): The phase model to be used for the
            interaction.
        is_reflective (bool, optional): Indicates whether the surface is
            reflective. Defaults to False.

    """

    def __init__(
        self,
        geometry: BaseGeometry,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
        phase_model: BasePhase,
        is_reflective: bool = False,
    ):
        super().__init__(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            is_reflective=is_reflective,
        )
        self.phase_model = phase_model

    def interact_real_rays(self, rays: RealRays) -> RealRays:
        """Interacts the real rays with the diffractive surface.

        Args:
            rays (RealRays): The real rays to be interacted with the surface.

        Returns:
            RealRays: The interacted real rays.

        """
        # find surface normals
        nx, ny, nz = self.geometry.surface_normal(rays)

        # get refractive indices
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # calculate phase
        L, M, N, opd = self.phase_model.phase_calc(rays, nx, ny, nz, n1, n2)

        # update rays
        rays.L = L
        rays.M = M
        rays.N = N
        rays.opd += opd
        rays.i *= self.phase_model.efficiency(rays)

        return rays

    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Raises a NotImplementedError for paraxial rays."""
        raise NotImplementedError(
            "Phase interaction is not supported for paraxial rays."
        )

    def flip(self):
        """Flips the interaction model."""
        self.material_pre, self.material_post = (
            self.material_post,
            self.material_pre,
        )

    def to_dict(self) -> dict:
        """Converts the interaction model to a dictionary."""
        interaction_dict = super().to_dict()
        interaction_dict.update({"phase_model": self.phase_model.to_dict()})
        return interaction_dict
