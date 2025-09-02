""" context for ray aiming strategies """
from typing import Optional, TYPE_CHECKING

from optiland.rays.aiming.strategy import RayAimingStrategy

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


class RayAimingContext:
    """Context for ray aiming strategies."""

    def __init__(self, strategy: Optional[RayAimingStrategy] = None):
        self._strategy = strategy

    def set_strategy(self, strategy: RayAimingStrategy):
        """Set the ray aiming strategy."""
        self._strategy = strategy

    def aim_ray(self, optic: "Optic", Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        """Aim a ray using the current strategy."""
        if self._strategy is None:
            raise ValueError("Ray aiming strategy not set.")
        return self._strategy.aim_ray(optic, Hx, Hy, Px, Py, wavelength)
