"""Base Zernike Module

This module contains the abstract base class for all zernike-related classes.

Kramer Harrison, 2025
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

import optiland.backend as be

_ZernikeIndex = np.dtype([("n", int), ("m", int)])


class BaseZernike(ABC):
    """
    Abstract base class for Zernike polynomials.

    Args:
        coeffs (array-like): The Zernike coefficients. Defaults to None.
        num_terms (int): the maximum number of terms. Only used if coeffs is None.
            Defaults to 36.
    """

    _indices_cache: ClassVar[np.ndarray | None] = None

    def __init__(self, coeffs=None, num_terms=36):
        self.coeffs = be.zeros(num_terms) if coeffs is None else coeffs
        self.indices = self._generate_indices(len(self.coeffs))

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

    @classmethod
    def _generate_indices(cls, n_indices: int) -> np.ndarray:
        """Generate the indices for Zernike terms.

        Args:
            n_indices (int): The number of indices to generate.

        Returns:
            list: List of tuples representing the indices (n, m) of the
                Zernike terms.
        """
        if cls._indices_cache is not None and len(cls._indices_cache) >= n_indices:
            return cls._indices_cache[:n_indices]

        numbers_present = np.full(n_indices + 1, False)
        # Set the first element to True if the notation is one-indexed
        numbers_present[0] = cls._index_to_number(0, 0) != 0
        number = []
        indices = []

        n = 0
        m = -n

        while not all(numbers_present):
            _number = cls._index_to_number(n, m)

            if _number is not None:
                number.append(_number)
                indices.append((n, m))

                if _number <= n_indices:
                    numbers_present[_number] = True

            if m == n:
                n += 1
                m = -n
            else:
                m += 1

        # sort indices according to Noll coefficient number
        indices_sorted = np.array(
            [element for _, element in sorted(zip(number, indices, strict=False))],
            dtype=_ZernikeIndex,
        )[:n_indices]

        cls._indices_cache = indices_sorted
        return indices_sorted

    @staticmethod
    @abstractmethod
    def _index_to_number(n: int, m: int) -> int | None:
        """Convert Zernike indices (n, m) to a coefficient number.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.

        Returns:
            int: The coefficient number corresponding to the Zernike indices.
        """
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
        """Calculate the radial term of the Zernike polynomial."""
        s_max = (n - abs(m)) // 2 + 1

        n = be.array(n)
        m = be.array(m)
        r = be.array(r)

        # Initialize value with correct backend
        value = be.zeros_like(r) if not isinstance(r, int | float) else 0.0

        for k in range(s_max):
            num = be.factorial(n - k)
            denom = (
                be.factorial(k)
                * be.factorial((n + m) // 2 - k)
                * be.factorial((n - m) // 2 - k)
            )
            coeff = (-1) ** k * num / denom
            term = coeff * (r ** (n - 2 * k))
            value = value + term

        return value

    def _azimuthal_term(self, m=0, phi=0):
        """Calculate the azimuthal term of the Zernike polynomial.

        Args:
            m (int): Azimuthal order of the Zernike term.
            phi (float): Azimuthal angle in radians.

        Returns:
            float: The calculated value of the azimuthal term.

        """
        m = be.array(m)
        phi = be.array(phi)

        if m >= 0:
            return be.cos(m * phi)
        return be.sin(be.abs(m) * phi)
