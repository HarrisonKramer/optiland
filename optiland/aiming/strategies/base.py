"""Base classes for ray aiming strategies.

This module defines the abstract base class for all ray aiming strategies.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland.optic.optic import Optic


class RayAimingStrategy(ABC):
    """Abstract base class for ray aiming strategies."""

    @abstractmethod
    def aim(
        self,
        optic: Optic,
        Hx: ArrayLike,
        Hy: ArrayLike,
        Px: ArrayLike,
        Py: ArrayLike,
        wavelength: float,
    ):
        """Given an optic, normalized field and pupil coordinates and wavelength,
        return the ray.

        Args:
            optic: The optic to aim the ray for.
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength: The wavelength of the ray in microns.

        Returns:
            The aimed ray(s).
        """
        raise NotImplementedError
