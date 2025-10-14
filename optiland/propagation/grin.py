"""Placeholder for Graded-Index (GRIN) propagation model."""
from typing import TYPE_CHECKING

from optiland.propagation.base import BasePropagationModel

if TYPE_CHECKING:
    from optiland.rays.real_rays import RealRays


class GrinPropagation(BasePropagationModel):
    """Placeholder for GRIN propagation.

    This model is not yet implemented and will raise an error if used.
    """

    def propagate(self, rays: 'RealRays', t: float) -> None:
        """Raises NotImplementedError.

        Args:
            rays: The rays object to be propagated.
            t: The distance to propagate.
        """
        raise NotImplementedError(
            "GRIN propagation is not yet implemented."
        )
