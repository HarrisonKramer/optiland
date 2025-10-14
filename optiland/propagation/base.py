"""Abstract base class for propagation models."""
import abc
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from optiland.rays.real_rays import RealRays


class BasePropagationModel(abc.ABC):
    """Abstract base class for all propagation models."""

    @abc.abstractmethod
    def propagate(self, rays: 'RealRays', t: float) -> None:
        """Propagates rays a distance t through the medium.

        This method modifies the ray state in-place.

        Args:
            rays: The rays object to be propagated.
            t: The distance to propagate along the z-axis.
        """
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the propagation model to a dictionary.

        By default, this only stores the class name, as the model is
        reconstructed and linked by the parent material during its own
        deserialization.
        """
        return {'class': self.__class__.__name__}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BasePropagationModel':
        """Deserializes a propagation model from a dictionary.

        Note: This is part of the common serialization interface but should
        not be used directly. Propagation models are instantiated and linked
        by their parent `BaseMaterial` during the material's deserialization
        process to correctly resolve circular dependencies.
        """
        raise NotImplementedError(
            "Propagation models must be deserialized by their parent Material "
            "to resolve dependencies."
        )
