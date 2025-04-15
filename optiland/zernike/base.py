"""Base Zernike Module

This module contains the abstract base class for all zernike-related classes.

Kramer Harrison, 2025
"""

import math
from abc import ABC, abstractmethod

import numpy as np


class BaseZernike(ABC):
    """
    Abstract base class for Zernike polynomials.

    Args:
        coeffs (array-like): The Zernike coefficients. Defaults to None.
        num_terms (int): the maximum number of terms. Only used if coeffs is None.
            Defaults to 36.
    """

    MAX_TERMS = 120

    def __init__(self, coeffs=None, num_terms=36):
        if coeffs is None:
            coeffs = np.zeros(num_terms, dtype=np.float64)
        if len(coeffs) > self.MAX_TERMS:
            raise ValueError("Number of coefficients is limited to 120.")

        self.coeffs = coeffs
        self.indices = self._generate_indices()

    def get_term(self, coeff=0, n=0, m=0, r=0, phi=0):
        """Calculate the Zernike term for given coefficients and parameters.

        Args:
            coeff (float): Coefficient value for the Zernike term.
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.
            r (float): Radial distance from the origin.
            phi (float): Azimuthal angle in radians.

        Returns:
            float: The calculated value of the Zernike term.

        """
        return (
            coeff
            * self._norm_constant(n, m)
            * self._radial_term(n, m, r)
            * self._azimuthal_term(m, phi)
        )

    def terms(self, r=0, phi=0):
        """Calculate the Zernike terms for given radial distance and azimuthal
        angle.

        Args:
            r (float): Radial distance from the origin.
            phi (float): Azimuthal angle in radians.

        Returns:
            list: List of calculated Zernike term values.

        """
        val = []
        for k, idx in enumerate(self.indices):
            n, m = idx
            try:
                val.append(self.get_term(self.coeffs[k], n, m, r, phi))
            except IndexError:
                break
        return val

    def poly(self, r=0, phi=0):
        """Calculate the Zernike polynomial for given radial distance and
        azimuthal angle.

        Args:
            r (float): Radial distance from the origin.
            phi (float): Azimuthal angle in radians.

        Returns:
            float: The calculated value of the Zernike polynomial.

        """
        return sum(self.terms(r, phi))

    @staticmethod
    @abstractmethod
    def _generate_indices():
        """Generate the indices for Zernike terms."""
        # pragma: no cover

    @abstractmethod
    def _norm_constant(n, m):
        """Calculate the normalization constant of the Zernike polynomial.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.

        Returns:
            float: The calculated value of the normalization constant.

        """
        # pragma: no cover

    def _radial_term(self, n, m, r):
        """Calculate the radial term of the Zernike polynomial.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.
            r (float): Radial distance from the origin.

        Returns:
            float: The calculated value of the radial term.

        """
        s_max = int((n - abs(m)) / 2 + 1)
        value = 0
        for k in range(s_max):
            value += (
                (-1) ** k
                * math.factorial(n - k)
                / (
                    math.factorial(k)
                    * math.factorial(int((n + m) / 2 - k))
                    * math.factorial(int((n - m) / 2 - k))
                )
                * r ** (n - 2 * k)
            )
        return value

    def _azimuthal_term(self, m=0, phi=0):
        """Calculate the azimuthal term of the Zernike polynomial.

        Args:
            m (int): Azimuthal order of the Zernike term.
            phi (float): Azimuthal angle in radians.

        Returns:
            float: The calculated value of the azimuthal term.

        """
        if m >= 0:
            return np.cos(m * phi)
        return np.sin(np.abs(m) * phi)
