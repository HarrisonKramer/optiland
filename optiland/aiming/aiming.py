"""Ray Aiming Module.

This module provides the RayAiming class, which is the main entry point for
the ray aiming functionality.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.aiming.factory import RayAimingStrategyFactory
from optiland.aiming.strategies.base import RayAimingStrategy

if TYPE_CHECKING:
    import optiland.v_math as np
    from optiland.aberrations import Aberrations
    from optiland.optic.optic import Optic
    from optiland.rays import RayBundle


class RayAiming:
    """Class for finding the entrance beam pupil for a given ray."""

    def __init__(
        self, optic: Optic, strategy: RayAimingStrategy | str = "paraxial", **kwargs
    ):
        """
        Initialize a new ray aiming instance.

        Args:
            optic: The optic to use for ray aiming.
            strategy: The aiming strategy to use. Can be a string identifier
                or an AimingStrategy instance. If a string, a strategy will be created
                using the RayAimingStrategyFactory. Defaults to 'paraxial'.
            **kwargs: Additional keyword arguments to pass to the strategy constructor
                if created from a string.
        """
        self.optic = optic
        if isinstance(strategy, str):
            self.strategy = RayAimingStrategyFactory.create_strategy(
                strategy, **kwargs
            )
        elif isinstance(strategy, RayAimingStrategy):
            self.strategy = strategy
        else:
            raise TypeError(
                "strategy must be a string or an RayAimingStrategy instance"
            )

    @property
    def chief_ray_aiming(self):
        return self.strategy

    def aim(self, rays: RayBundle) -> RayBundle:
        """
        Aim the given rays through the optic.

        Args:
            rays: The rays to aim.

        Returns:
            The aimed rays.
        """
        return self.strategy.aim(rays)

    def get_reference_ray_from_field_point(
        self, field_point: np.ndarray, wavelength_index: int
    ) -> Aberrations:
        """
        Get the reference ray for a given field point.

        Args:
            field_point: The field point to use.
            wavelength_index: The wavelength index to use.

        Returns:
            The reference ray.
        """
        return self.strategy.get_reference_ray_from_field_point(
            field_point, wavelength_index
        )
