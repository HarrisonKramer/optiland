"""Ideal Material

This module contains the IdealMaterial class, which represents an ideal
material with a fixed refractive index and extinction coefficient for all
wavelengths.

from typing import Any, TYPE_CHECKING
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.materials.base import BaseMaterial
from optiland.propagation.base import BasePropagationModel

if TYPE_CHECKING:
    from optiland.propagation.base import BasePropagationModel


class IdealMaterial(BaseMaterial):
    """Represents an ideal material with a fixed refractive index and extinction
    coefficient for all wavelengths.

    By default, the refractive index of this material is considered absolute
    and does not change with the environment.

    Args:
        n (float): The refractive index of the material.
        k (float): The extinction coefficient of the material. Defaults to 0.0.
        relative_to_environment (bool): If True, the refractive index is
            treated as relative to the environment. Defaults to False.

    Attributes:
        n_val (float): The refractive index of the material.
        k_val (float): The extinction coefficient of the material.
        relative_to_environment (bool): Flag indicating if the index is relative.
    """

    def __init__(
        self,
        n: float,
        k: float = 0.0,
        relative_to_environment: bool = False,
        propagation_model: BasePropagationModel | None = None,
    ):
        super().__init__(propagation_model)
        self.n_val = n
        self.k_val = k
        self.relative_to_environment = relative_to_environment

    def n(self, wavelength: float | be.ndarray, **kwargs: Any) -> float | be.ndarray:
        """
        Calculates the refractive index of the material.

        If `relative_to_environment` is False, this method returns the constant
        refractive index `n_val`. Otherwise, it calculates the index relative
        to the current environment using the base class implementation.
        """
        if not self.relative_to_environment:
            if be.is_array_like(wavelength) and be.size(wavelength) > 1:
                return be.full_like(wavelength, self.n_val)
            return self.n_val
        return super().n(wavelength, **kwargs)

    def _calculate_absolute_n(
        self, wavelength: float | be.ndarray, **kwargs: Any
    ) -> float | be.ndarray:
        """Returns the absolute refractive index of the material."""
        if be.is_array_like(wavelength) and be.size(wavelength) > 1:
            return be.full_like(wavelength, self.n_val)
        return self.n_val

    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs: Any
    ) -> float | be.ndarray:
        """Returns the constant extinction coefficient of the material."""
        if be.is_array_like(wavelength) and be.size(wavelength) > 1:
            return be.full_like(wavelength, self.k_val)
        return self.k_val

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the material."""
        material_dict = super().to_dict()
        material_dict.update(
            {
                "n": self.n_val,
                "k": self.k_val,
                "relative_to_environment": self.relative_to_environment,
            }
        )
        return material_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdealMaterial:
        """Creates an IdealMaterial instance from a dictionary representation."""
        return cls(
            data["n"],
            data.get("k", 0.0),
            data.get("relative_to_environment", False),
        )
