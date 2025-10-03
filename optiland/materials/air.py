"""Material Air Module

This module defines the Air material class, which represents air as a
material whose refractive index depends on wavelength and environmental
conditions. The refractive index is calculated using various established
empirical models.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.environment.air_index import refractive_index_air
from optiland.materials.base import BaseMaterial

if TYPE_CHECKING:
    from optiland.environment.conditions import EnvironmentalConditions


class Air(BaseMaterial):
    """
    Represents air as a material whose refractive index is dependent
    on wavelength and environmental conditions.

    Args:
        conditions (EnvironmentalConditions): The environmental conditions
            (temperature, pressure) affecting the refractive index of air.
        model (str): The empirical model to use for refractive index
            calculation. Options include "ciddor", "edlen", "birch_downs",
            and "kohlrausch". Default is "ciddor".
    """

    def __init__(self, conditions: EnvironmentalConditions, model: str = "ciddor"):
        super().__init__()
        self.conditions = conditions
        self.model = model

    def _calculate_absolute_n(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """
        Calculates the absolute refractive index of air using the specified model.
        """
        # TODO: Handle array inputs more efficiently
        if be.is_array_like(wavelength):
            return be.array(
                [
                    refractive_index_air(w, self.conditions, self.model)
                    for w in be.to_numpy(wavelength).flatten()
                ]
            ).reshape(wavelength.shape)
        return refractive_index_air(wavelength, self.conditions, self.model)

    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Extinction coefficient of air is assumed negligible."""
        if be.is_array_like(wavelength):
            return be.zeros_like(wavelength)
        return 0.0
