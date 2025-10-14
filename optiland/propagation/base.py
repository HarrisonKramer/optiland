"""Abstract base class for propagation models.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays.real_rays import RealRays


class BasePropagationModel(abc.ABC):
    """Abstract base class for all propagation models."""

    @abc.abstractmethod
    def propagate(self, rays: RealRays, t: float) -> None:
        """Propagates rays a distance t through the medium.

        This method modifies the ray state in-place.

        Args:
            rays: The rays object to be propagated.
            t: The distance to propagate along the z-axis.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serializes the propagation model to a dictionary.

        By default, this only stores the class name, as the model is
        reconstructed and linked by the parent material during its own
        deserialization.
        """
        return {"class": self.__class__.__name__}

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls, d: dict[str, Any], material: BaseMaterial
    ) -> BasePropagationModel:
        """Deserializes a propagation model from a dictionary.

        Note: This method should be called by the parent `BaseMaterial`
        during its deserialization process to correctly resolve circular
        dependencies.

        Args:
            d: A dictionary containing the serialized data.
            material: The parent material instance.
        """
        raise NotImplementedError
