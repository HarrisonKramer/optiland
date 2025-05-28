"""Abbe Material

This module defines a material based on the refractive index at the Fraunhofer
d-line (587.56 nm) and the Abbe number. The refractive index is based on a
polynomial fit to glass data from the Schott catalog. The extinction
coefficient is ignored in this model and is always set to zero.

Kramer Harrison, 2024
"""

from importlib import resources

import optiland.backend as be
from optiland.materials.base import BaseMaterial


class AbbeMaterial(BaseMaterial):
    """Represents a material based on the refractive index at the Fraunhofer
    d-line (587.56 nm) and the Abbe number. The refractive index is based on
    a polynomial fit to glass data from the Schott catalog. The extinction
    coefficient is ignored in this model and is always set to zero.

    Attributes:
        index (float): The refractive index of the material at 587.56 nm.
        abbe (float): The Abbey number of the material.

    """

    def __init__(self, n, abbe):
        self.index = be.array([n])
        self.abbe = be.array([abbe])
        self._p = self._get_coefficients()

    def n(self, wavelength):
        """Returns the refractive index of the material.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            be.ndarray: The refractive index of the material at the given
            wavelength(s).

        """
        wavelength = be.array(wavelength)
        if be.any(wavelength < 0.380) or be.any(wavelength > 0.750):
            raise ValueError("Wavelength out of range for this model.")
        return be.atleast_1d(be.polyval(self._p, wavelength))

    def k(self, wavelength):
        """Returns the extinction coefficient of the material.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float or be.ndarray: The extinction coefficient of the material, which
            is always 0 for this model. Returns a scalar 0 if wavelength is scalar,
            otherwise an array of zeros.

        """
        return 0

    def _get_coefficients(self):
        """Returns the polynomial coefficients for the refractive index model.

        These coefficients are used in `be.polyval` to calculate the refractive
        index at a given wavelength.

        Returns:
            be.ndarray: A 1D array of polynomial coefficients.

        """
        # Polynomial fit to the refractive index data
        X_poly = be.ravel(
            be.array(
                [
                    self.index,
                    self.abbe,
                    self.index**2,
                    self.abbe**2,
                    self.index**3,
                    self.abbe**3,
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

    def to_dict(self):
        """Returns a dictionary representation of the material.

        Returns:
            dict: The dictionary representation of the material.

        """
        material_dict = super().to_dict()
        material_dict.update({"index": float(self.index), "abbe": float(self.abbe)})
        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            AbbeMaterial: The material object.

        """
        required_keys = ["index", "abbe"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        return cls(data["index"], data["abbe"])
