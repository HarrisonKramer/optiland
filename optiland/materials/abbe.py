import os
import numpy as np
from optiland.materials.base import BaseMaterial


class AbbeMaterial(BaseMaterial):
    """
    Represents a material based on the refractive index at the Fraunhofer
    d-line (587.56 nm) and the Abbe number. The refractive index is based on
    a polynomial fit to glass data from the Schott catalog. The absorption
    coefficient is ignored in this model and is always set to zero.

    Attributes:
        index (float): The refractive index of the material at 587.56 nm.
        abbe (float): The Abbey number of the material.
    """

    def __init__(self, n, abbe):
        self.index = n
        self.abbe = abbe
        self._p = self._get_coefficients()

    def n(self, wavelength):
        """
        Returns the refractive index of the material.

        Args:
            wavelength (float): The wavelength of light in microns.

        Returns:
            float: The refractive index of the material.
        """
        return np.polyval(self._p, wavelength)

    def k(self, wavelength):
        """
        Returns the absorption coefficient of the material.

        Args:
            wavelength (float): The wavelength of light in microns.

        Returns:
            float: The absorption coefficient of the material.
        """
        return 0

    def _get_coefficients(self):
        """
        Returns the polynomial coefficients for the refractive index.

        Returns:
            numpy.ndarray: The polynomial coefficients.
        """
        # Polynomial fit to the refractive index data
        X = np.array([self.index, self.abbe])
        X_poly = np.hstack([X**i for i in range(1, 4)])

        # File contains fit coefficients
        coefficients_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../../database/glass_model_coefficients.npy'
            )

        coefficients = np.load(coefficients_file)
        return X_poly @ coefficients
