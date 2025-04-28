"""Standard Zernike Module

This module contains the ZernikeStandard class, which defines Zernike polynomials
based on the "standard" indexing scheme.

References:
    1. https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices
    2. Thibos LN, Applegate RA, Schwiegerling JT, Webb R; VSIA Standards
       Taskforce Members. Vision science and its applications. Standards
       for reporting the optical aberrations of eyes. J Refract Surg. 2002
       Sep-Oct;18(5):S652-60. doi: 10.3928/1081-597X-20020901-30. PMID:
       12361175.

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.zernike.base import BaseZernike


class ZernikeStandard(BaseZernike):
    """OSA/ANSI Standard Zernike

    This class represents the OSA/ANSI Standard Zernike polynomials.
    It provides methods to calculate the Zernike terms, Zernike polynomials,
    and other related functions.

    Args:
        coeffs (array-like): The Zernike coefficients. Defaults to None.
        num_terms (int): the maximum number of terms. Only used if coeffs is None.
            Defaults to 36.

    References:
        1. https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/
           ANSI_standard_indices
        2. Thibos LN, Applegate RA, Schwiegerling JT, Webb R; VSIA Standards
           Taskforce Members. Vision science and its applications. Standards
           for reporting the optical aberrations of eyes. J Refract Surg. 2002
           Sep-Oct;18(5):S652-60. doi: 10.3928/1081-597X-20020901-30. PMID:
           12361175.

    """

    def __init__(self, coeffs=None, num_terms=36):
        super().__init__(coeffs, num_terms)

    @staticmethod
    def _generate_indices():
        """Generate the indices for the Zernike terms.

        Returns:
            list: List of tuples representing the indices (n, m) of the
                Zernike terms.

        """
        indices = []
        for n in range(15):
            for m in range(-n, n + 1):
                if (n - m) % 2 == 0:
                    indices.append((n, m))
        return indices

    @staticmethod
    def _norm_constant(n=0, m=0):
        """Calculate the normalization constant of the Zernike polynomial.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.

        Returns:
            float: The calculated value of the normalization constant.

        """
        return be.sqrt(be.array((2 * n + 2) / (1 + (m == 0))))
