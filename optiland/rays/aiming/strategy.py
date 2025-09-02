""" abstract base class for ray aiming strategies """
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


class RayAimingStrategy(ABC):
    """Abstract base class for ray aiming strategies."""

    @abstractmethod
    def aim_ray(self, optic: "Optic", Hx: float, Hy: float, Px: float, Py: float, wavelength: float):
        """given an optic, normalized field and pupil coordinates and wavelength, return the ray"""
        raise NotImplementedError
