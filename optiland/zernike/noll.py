"""Noll Zernike Module

This module contains the NollZernike class, which defines Zernike polynomials
based on the "Noll" indexing scheme. Note that the Noll notation is used for the
"Zernike Standard Coefficients" in Ansys Zemax OpticStudio.

References:
    1. https://en.wikipedia.org/wiki/
       Zernike_polynomials#Noll's_sequential_indices
    2. Noll, R. J. (1976). "Zernike polynomials and atmospheric
       turbulence". J. Opt. Soc. Am. 66 (3): 207

Kramer Harrison, 2025
"""

import numpy as np

from optiland.zernike.base import BaseZernike


class ZernikeNoll(BaseZernike):
    """Zernike Coefficients - Noll Standard

    This class represents Zernike Noll Coefficients. It is a subclass of the
    BaseZernike class. Note that the Noll notation is used for the
    "Zernike Standard Coefficients" in Ansys Zemax OpticStudio.

    Args:
        coeffs (list): the coefficient list for the Zernike polynomials.
            Defaults to all zeros (36 elements total)

    References:
        1. https://en.wikipedia.org/wiki/
           Zernike_polynomials#Noll's_sequential_indices
        2. Noll, R. J. (1976). "Zernike polynomials and atmospheric
           turbulence". J. Opt. Soc. Am. 66 (3): 207

    """

    def __init__(self, coeffs=None):
        if coeffs is None:
            coeffs = [0 for _ in range(36)]
        super().__init__(coeffs)

    @staticmethod
    def _norm_constant(n=0, m=0):
        """Calculate the normalization constant for a given Zernike polynomial.

        Args:
            n (int): The radial order of the Zernike polynomial.
            m (int): The azimuthal order of the Zernike polynomial.

        Returns:
            float: The normalization constant for the Zernike polynomial.

        """
        if m == 0:
            return np.sqrt(n + 1)
        return np.sqrt(2 * n + 2)

    @staticmethod
    def _generate_indices():
        """Generate the indices for the Zernike terms.

        Returns:
            list: List of tuples representing the indices (n, m) of the
                Zernike terms.

        """
        number = []
        indices = []
        for n in range(15):
            for m in range(-n, n + 1):
                if (n - m) % 2 == 0:
                    mod = n % 4
                    if (m > 0 and mod <= 1) or (m < 0 and mod >= 2):
                        c = 0
                    elif (m >= 0 and mod >= 2) or (m <= 0 and mod <= 1):
                        c = 1
                    number.append(n * (n + 1) / 2 + np.abs(m) + c)
                    indices.append((n, m))

        # sort indices according to fringe coefficient number
        indices_sorted = [element for _, element in sorted(zip(number, indices))]

        return indices_sorted
