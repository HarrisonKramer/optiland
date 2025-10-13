"""Abbe Material

This module defines a material based on the refractive index at the Fraunhofer
d-line (587.56 nm) and the Abbe number. The refractive index is based on a
polynomial fit to glass data from the Schott catalog. The extinction
coefficient is ignored in this model and is always set to zero.

Kramer Harrison, 2024
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import optiland.backend as be
from optiland.environment.air_index import refractive_index_air
from optiland.environment.conditions import EnvironmentalConditions
from optiland.materials.base import BaseMaterial


class AbbeMaterial(BaseMaterial):
    """Represents a material based on the refractive index at the Fraunhofer
    d-line (nd, 587.56 nm) and the Abbe number (Vd).

    The refractive index is based on a polynomial fit to glass data from the
    Schott catalog. The extinction coefficient is ignored in this model and is
    always set to zero. This model assumes the provided nd and Vd are relative
    to standard air.

    Args:
        n (float): The refractive index of the material at 587.56 nm.
        abbe (float): The Abbe number of the material.

    Attributes:
        n_val (float): The refractive index of the material at 587.56 nm.
        abbe_val (float): The Abbe number of the material.
    """

    def __init__(self, n: float, abbe: float):
        super().__init__()
        self.n_val = n
        self.abbe_val = abbe
        self._p = self._get_coefficients()
        # Standard conditions for the air reference of the glass model
        self._reference_conditions = EnvironmentalConditions(
            temperature=20.0, pressure=101325.0, relative_humidity=0.0
        )

    def _calculate_absolute_n(
        self, wavelength: float | be.ndarray, **kwargs: Any
    ) -> float | be.ndarray:
        """
        Calculates the absolute refractive index.

        The calculation is based on a polynomial fit that yields a refractive
        index relative to air, which is then converted to an absolute index.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            float | be.ndarray: The absolute refractive index of the material.
        """
        if be.is_array_like(wavelength):
            if be.any(wavelength < 0.380) or be.any(wavelength > 0.750):
                raise ValueError("Wavelength out of range for this model.")
        else:
            if wavelength < 0.380 or wavelength > 0.750:
                raise ValueError("Wavelength out of range for this model.")

        # The polynomial model gives the index relative to standard air
        relative_n = be.polyval(self._p, be.asarray(wavelength))

        # Convert to absolute index by multiplying by the index of standard air
        n_air_reference = refractive_index_air(
            wavelength, self._reference_conditions, model="kohlrausch"
        )
        absolute_n = relative_n * n_air_reference

        return be.atleast_1d(absolute_n)

    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs: Any
    ) -> float | be.ndarray:
        """Returns the extinction coefficient of the material (always zero).

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            float or be.ndarray: The extinction coefficient, which is always 0.
        """
        if be.is_array_like(wavelength) and be.size(wavelength) > 1:
            return be.zeros_like(wavelength)
        return 0.0

    def _get_coefficients(self) -> be.ndarray:
        """Returns the polynomial coefficients for the refractive index model.

        These coefficients are derived from a pre-computed fit and are used in
        `be.polyval` to calculate the refractive index at a given wavelength.

        Returns:
            be.ndarray: A 1D array of polynomial coefficients.
        """
        # Polynomial fit to the refractive index data based on n_val and abbe_val
        X_poly = be.ravel(
            be.array(
                [
                    self.n_val,
                    self.abbe_val,
                    self.n_val**2,
                    self.abbe_val**2,
                    self.n_val**3,
                    self.abbe_val**3,
                ]
            )
        )

        # File contains fit coefficients
        coefficients_file = str(
            resources.files("optiland.database").joinpath(
                "glass_model_coefficients.npy",
            ),
        )
        coefficients = be.load(coefficients_file)
        return be.matmul(X_poly, coefficients)

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the material.

        Returns:
            dict: The dictionary representation of the material.
        """
        material_dict = super().to_dict()
        material_dict.update({"n": self.n_val, "abbe": self.abbe_val})
        return material_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AbbeMaterial:
        """Creates a material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            AbbeMaterial: The material object.
        """
        required_keys = ["n", "abbe"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        return cls(data["n"], data["abbe"])
