"""Interaction Model Factory

This module contains the InteractionModelFactory class, which is used to create
interaction model objects based on the given parameters in an OCP-compliant way.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.interactions.diffractive_model import DiffractiveInteractionModel
from optiland.interactions.phase_interaction_model import PhaseInteractionModel
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.coatings import BaseCoating
    from optiland.interactions.base import BaseInteractionModel
    from optiland.scatter import BaseBSDF
    from optiland.surfaces import Surface


class InteractionModelFactory:
    """A factory for creating interaction model objects using a registry."""

    def __init__(self):
        self._builders: dict[str, Callable[..., BaseInteractionModel]] = {
            "refractive_reflective": self._build_refractive_reflective_model,
            "thin_lens": self._build_thin_lens_model,
            "diffractive": self._build_diffractive_model,
            "phase": self._build_phase_model,
        }

    def register_builder(
        self, interaction_type: str, builder: Callable[..., BaseInteractionModel]
    ):
        """Registers a new interaction model builder."""
        self._builders[interaction_type] = builder

    def create(
        self,
        parent_surface: Surface | None,
        interaction_type: str,
        is_reflective: bool,
        coating: BaseCoating | None,
        bsdf: BaseBSDF | None,
        **kwargs,
    ) -> BaseInteractionModel:
        """Creates an interaction model using a registered builder.

        Args:
            parent_surface: The parent surface of the interaction model.
            interaction_type: The type of interaction model to create.
            is_reflective: Indicates whether the surface is reflective.
            coating: The coating of the surface.
            bsdf: The BSDF of the surface.
            **kwargs: Additional keyword arguments for the specific model.

        Returns:
            The created interaction model object.

        Raises:
            ValueError: If the interaction_type is unknown.
        """
        builder = self._builders.get(interaction_type)
        if not builder:
            raise ValueError(f"Unknown interaction_type: {interaction_type}")

        common_args = {
            "parent_surface": parent_surface,
            "is_reflective": is_reflective,
            "coating": coating,
            "bsdf": bsdf,
        }
        return builder(**common_args, **kwargs)

    @staticmethod
    def _build_refractive_reflective_model(**kwargs) -> RefractiveReflectiveModel:
        return RefractiveReflectiveModel(**kwargs)

    @staticmethod
    def _build_thin_lens_model(**kwargs) -> ThinLensInteractionModel:
        if "focal_length" not in kwargs:
            raise ValueError("Focal length is required for thin lens.")
        return ThinLensInteractionModel(**kwargs)

    @staticmethod
    def _build_diffractive_model(**kwargs) -> DiffractiveInteractionModel:
        return DiffractiveInteractionModel(**kwargs)

    @staticmethod
    def _build_phase_model(**kwargs) -> PhaseInteractionModel:
        if "phase_profile" not in kwargs:
            raise ValueError("phase_profile is required for phase interaction.")
        return PhaseInteractionModel(**kwargs)
