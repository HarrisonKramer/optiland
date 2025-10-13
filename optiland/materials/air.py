"""Material Air Module

This module defines the Air material class, which represents air as a
material whose refractive index depends on wavelength and environmental
conditions. The refractive index is calculated using various established
empirical models.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.materials.base import BaseMaterial

if TYPE_CHECKING:
    from optiland.environment.conditions import EnvironmentalConditions


class Air(BaseMaterial):
    """Represents air as a material whose refractive index is dependent
    on wavelength and environmental conditions.

    Args:
        conditions (EnvironmentalConditions): The environmental conditions
            (temperature, pressure) affecting the refractive index of air.
        model (str): The empirical model to use for refractive index
            calculation. Options include "ciddor", "edlen", "birch_downs",
            and "kohlrausch". Defaults to "kohlrausch".

    Attributes:
        conditions (EnvironmentalConditions): The environmental conditions.
        model (str): The refractive index model name.
    """

    def __init__(self, conditions: EnvironmentalConditions, model: str = "kohlrausch"):
        super().__init__()
        self.conditions = conditions
        self.model = model

    def _calculate_absolute_n(
        self, wavelength: float | be.ndarray, **kwargs: Any
    ) -> float | be.ndarray:
        """Calculates the absolute refractive index of air using the specified model.

        Args:
            wavelength (float | be.ndarray): The wavelength(s) in microns.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            float | be.ndarray: The absolute refractive index of air.
        """
        from optiland.environment.air_index import refractive_index_air

        return refractive_index_air(wavelength, self.conditions, self.model)

    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs: Any
    ) -> float | be.ndarray:
        """Extinction coefficient of air is assumed to be negligible (zero).

        Args:
            wavelength (float | be.ndarray): The wavelength(s) in microns.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            float | be.ndarray: The extinction coefficient, which is always 0.
        """
        if be.is_array_like(wavelength) and be.size(wavelength) > 1:
            return be.zeros_like(wavelength)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the Air material.

        Returns:
            dict: A dictionary containing the material's type, conditions, and model.
        """
        material_dict = super().to_dict()
        material_dict.update(
            {
                "conditions": self.conditions.to_dict(),
                "model": self.model,
            }
        )
        return material_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Air:
        """Creates an Air material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            Air: The deserialized Air material instance.
        """
        from optiland.environment.conditions import EnvironmentalConditions

        conditions = EnvironmentalConditions.from_dict(data["conditions"])
        return cls(conditions, data.get("model", "kohlrausch"))
