"""Interaction Model Factory

This module contains the InteractionModelFactory class, which is used to create
interaction model objects based on the given parameters.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.interactions.diffractive_model import DiffractiveInteractionModel
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel

if TYPE_CHECKING:
    # pragma: no cover
    from optiland.coatings import BaseCoating
    from optiland.geometries.base import BaseGeometry
    from optiland.interactions.base import BaseInteractionModel
    from optiland.materials.base import BaseMaterial
    from optiland.scatter import BaseBSDF


class InteractionModelFactory:
    """A factory class for creating interaction model objects."""

    def create(
        self,
        interaction_type: str,
        geometry: BaseGeometry,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
        is_reflective: bool,
        coating: BaseCoating | None,
        bsdf: BaseBSDF | None,
        focal_length: float | None = None,
    ) -> BaseInteractionModel:
        """Creates an interaction model object based on the given parameters.

        Args:
            interaction_type (str): The type of interaction model to create.
            geometry (BaseGeometry): The geometry of the surface.
            material_pre (BaseMaterial): The material before the surface.
            material_post (BaseMaterial): The material after the surface.
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
                geometry=geometry,
                material_pre=material_pre,
                material_post=material_post,
                is_reflective=is_reflective,
                coating=coating,
                bsdf=bsdf,
            )
        elif interaction_type == "thin_lens":
            if focal_length is None:
                raise ValueError("Focal length is required for thin lens.")
            return ThinLensInteractionModel(
                focal_length=focal_length,
                geometry=geometry,
                material_pre=material_pre,
                material_post=material_post,
                is_reflective=is_reflective,
                coating=coating,
                bsdf=bsdf,
            )
        elif interaction_type == "diffractive":
            return DiffractiveInteractionModel(
                geometry=geometry,
                material_pre=material_pre,
                material_post=material_post,
                is_reflective=is_reflective,
                coating=coating,
                bsdf=bsdf,
            )
        else:
            raise ValueError(f"Unknown interaction_type: {interaction_type}")
