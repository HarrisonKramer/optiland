"""Interaction Model Factory

This module contains the InteractionModelFactory class, which is used to create
interaction model objects based on the given parameters.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.interactions.diffractive_model import DiffractiveInteractionModel
from optiland.interactions.phase_interaction_model import PhaseInteractionModel
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel

if TYPE_CHECKING:
    # pragma: no cover
    from optiland.coatings import BaseCoating
    from optiland.interactions.base import BaseInteractionModel
    from optiland.scatter import BaseBSDF
    from optiland.surfaces import Surface


class InteractionModelFactory:
    """A factory class for creating interaction model objects."""

    def create(
        self,
        parent_surface: Surface | None,
        interaction_type: str,
        is_reflective: bool,
        coating: BaseCoating | None,
        bsdf: BaseBSDF | None,
        **kwargs,
    ) -> BaseInteractionModel:
        """Creates an interaction model object based on the given parameters.

        Args:
            interaction_type (str): The type of interaction model to create.
            is_reflective (bool): Indicates whether the surface is reflective.
            coating (Optional[BaseCoating]): The coating of the surface.
            bsdf (Optional[BaseBSDF]): The BSDF of the surface.
            focal_length (Optional[float]): The focal length of the surface.

        Returns:
            BaseInteractionModel: The created interaction model object.

        Raises:
            ValueError: If the interaction_type is unknown.
        """
        if interaction_type == "refractive_reflective":
            return RefractiveReflectiveModel(
                parent_surface=parent_surface,
                is_reflective=is_reflective,
                coating=coating,
                bsdf=bsdf,
            )
        elif interaction_type == "thin_lens":
            focal_length = kwargs.get("focal_length")
            if focal_length is None:
                raise ValueError("Focal length is required for thin lens.")
            return ThinLensInteractionModel(
                parent_surface=parent_surface,
                focal_length=focal_length,
                is_reflective=is_reflective,
                coating=coating,
                bsdf=bsdf,
            )
        elif interaction_type == "diffractive":
            return DiffractiveInteractionModel(
                parent_surface=parent_surface,
                is_reflective=is_reflective,
                coating=coating,
                bsdf=bsdf,
            )
        elif interaction_type == "phase":
            phase_profile = kwargs.get("phase_profile")
            if phase_profile is None:
                raise ValueError("phase_profile is required for phase interaction.")
            return PhaseInteractionModel(
                parent_surface=parent_surface,
                phase_profile=phase_profile,
                is_reflective=is_reflective,
                coating=coating,
                bsdf=bsdf,
            )
        else:
            raise ValueError(f"Unknown interaction_type: {interaction_type}")
