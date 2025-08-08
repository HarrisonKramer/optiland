"""Fringe Zernike Module

This module contains the FringeZernike class, which defines Zernike polynomials
based on the "fringe", or "University of Arizona", indexing scheme.

References:
    1. https://en.wikipedia.org/wiki/Zernike_polynomials#Fringe/
       University_of_Arizona_indices

Kramer Harrison, 2025
"""

import numpy as np

import optiland.backend as be
from optiland.zernike.base import BaseZernike


class ZernikeFringe(BaseZernike):
    """Zernike Fringe Coefficients

    This class represents Zernike Fringe Coefficients. It is a subclass of the
    BaseZernike class.

    Args:
        coeffs (list): the coefficient list for the Zernike polynomials.
            Defaults to all zeros (36 elements total)

    References:
        1. https://en.wikipedia.org/wiki/Zernike_polynomials#Fringe/
           University_of_Arizona_indices

    """
    
    @staticmethod
    def _norm_constant(n=0, m=0):
        """Calculate the normalization constant for a given Zernike polynomial.
        Note that this is 1 for all terms for Zernike Fringe polynomials.

        Args:
            n (int): The radial order of the Zernike polynomial.
            m (int): The azimuthal order of the Zernike polynomial.

        Returns:
            float: The normalization constant for the Zernike polynomial.

        """
        return be.array(1)

    @staticmethod
    def _generate_indices(n_indices: int):
        """Generate the indices for the Zernike terms.

        Returns:
            list: List of tuples representing the indices (n, m) of the
                Zernike terms.

        """
        numbers_present = np.full(n_indices + 1, False)
        numbers_present[0] = True

        number = []
        indices = []

        n = 0

        # Iterate until all numbers up to n_indices are present
        while not np.all(numbers_present):
            for m in range(-n, n + 1):
                if (n - m) % 2 == 0:
                    _number = int(
                        (1 + (n + np.abs(m)) / 2) ** 2
                        - 2 * np.abs(m)
                        + (1 - np.sign(m)) / 2
                    )

                    number.append(_number)

                    if _number <= n_indices:
                        numbers_present[_number] = True

                    indices.append((n, m))

            n += 1

        # sort indices according to fringe coefficient number
        indices_sorted = [
            element for _, element in sorted(zip(number, indices, strict=False))
        ]

        return indices_sorted[:n_indices]
