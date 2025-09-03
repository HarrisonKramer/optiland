"""Ray Aiming Module.

This module provides the RayAiming class, which is the main entry point for
the ray aiming functionality.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland.aiming.strategies.base import RayAimingStrategy
    from optiland.optic.optic import Optic


class RayAiming:
    """A class to manage and execute ray aiming strategies.

    This class holds a ray aiming strategy and uses it to aim rays for a given
    optic. It provides a simple interface to the user, who can set the
    strategy and then call the aim_ray method.

    Attributes:
        _strategy (RayAimingStrategy | None): The ray aiming strategy to use.
    """

    def __init__(self, strategy: RayAimingStrategy | None = None) -> None:
        """Initializes the RayAiming class.

        Args:
            strategy: The ray aiming strategy to use. If None, a strategy must
                be set before aiming rays.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: RayAimingStrategy) -> None:
        """Sets the ray aiming strategy.

        Args:
            strategy: The ray aiming strategy to use.
        """
        self._strategy = strategy

    def aim_ray(
        self,
        optic: Optic,
        Hx: ArrayLike,
        Hy: ArrayLike,
        Px: ArrayLike,
        Py: ArrayLike,
        wavelength: float,
    ):
        """Aims a ray using the current strategy.

        Args:
            optic: The optic to aim the ray for.
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength: The wavelength of the ray in microns.

        Returns:
            The aimed ray(s).

        Raises:
            ValueError: If the ray aiming strategy is not set.
        """
        if self._strategy is None:
            raise ValueError("Ray aiming strategy not set.")
        return self._strategy.aim_ray(optic, Hx, Hy, Px, Py, wavelength)
