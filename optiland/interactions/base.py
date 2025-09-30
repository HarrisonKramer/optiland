"""Base Interaction Model

Defines the abstract base class for ray-surface interaction models.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pragma: no cover
    from optiland.coatings import BaseCoating
    from optiland.geometries import BaseGeometry
    from optiland.materials import BaseMaterial
    from optiland.rays import ParaxialRays, RealRays
    from optiland.scatter import BaseBSDF


class BaseInteractionModel(ABC):
    """Abstract base class for ray-surface interaction models."""

    _registry = {}

    def __init__(
        self,
        geometry: BaseGeometry,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
        is_reflective: bool,
        coating: BaseCoating | None = None,
        bsdf: BaseBSDF | None = None,
    ):
        self.geometry = geometry
        self.material_pre = material_pre
        self.material_post = material_post
        self.is_reflective = is_reflective
        self.coating = coating
        self.bsdf = bsdf

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseInteractionModel._registry[cls.__name__] = cls

    @abstractmethod
    def interact_real_rays(self, rays: RealRays) -> RealRays:
        """Interact with real rays."""
        pass  # pragma: no cover

    @abstractmethod
    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Interact with paraxial rays."""
        pass  # pragma: no cover

    @abstractmethod
    def flip(self):
        """Flip the interaction model."""
        pass  # pragma: no cover

    def to_dict(self):
        """Returns a dictionary representation of the interaction model."""
        return {
            "type": self.__class__.__name__,
            "is_reflective": self.is_reflective,
            "coating": self.coating.to_dict() if self.coating else None,
            "bsdf": self.bsdf.to_dict() if self.bsdf else None,
        }

    @classmethod
    def from_dict(cls, data, geometry, material_pre, material_post):
        """Creates an interaction model from a dictionary representation."""
        from optiland.coatings import BaseCoating
        from optiland.scatter import BaseBSDF

        interaction_type = data["type"]
        subclass = cls._registry.get(interaction_type)
        if subclass is None:
            raise ValueError(f"Unknown interaction model type: {interaction_type}")

        # Remove 'type' from data to avoid passing it to the constructor
        init_data = data.copy()
        init_data.pop("type")

        if "coating" in init_data and init_data["coating"] is not None:
            init_data["coating"] = BaseCoating.from_dict(init_data["coating"])
        if "bsdf" in init_data and init_data["bsdf"] is not None:
            init_data["bsdf"] = BaseBSDF.from_dict(init_data["bsdf"])

        return subclass(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            **init_data,
        )

    def _apply_coating_and_bsdf(
        self, rays: RealRays, nx: float, ny: float, nz: float
    ) -> RealRays:
        """Apply coating and BSDF to the rays."""
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx, ny, nz)

        if self.coating:
            rays = self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=nx,
                ny=ny,
                nz=nz,
            )
        else:
            rays.update()
        return rays
