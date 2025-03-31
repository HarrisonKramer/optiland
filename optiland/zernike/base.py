from abc import ABC, abstractmethod

import numpy as np
from scipy.special import comb


class BaseZernike(ABC):
    """
    Abstract base class for Zernike polynomials.
    """

    MAX_TERMS = 120

    def __init__(self, coeffs=None, max_terms=36):
        if coeffs is None:
            coeffs = np.zeros(max_terms, dtype=np.float64)
        if len(coeffs) > self.MAX_TERMS:
            raise ValueError("Number of coefficients is limited to 120.")

        self.coeffs = coeffs
        self.indices = self.generate_indices()

    @abstractmethod
    def generate_indices(self):
        """Generate the indices for Zernike terms."""
        pass

    def _radial_term(self, n, m, r):
        k = np.arange((n - abs(m)) // 2 + 1)
        terms = (
            (-1) ** k
            * comb(n - k, k)
            * comb((n + m) // 2 - k, (n - m) // 2 - k)
            * r ** (n - 2 * k)
        )
        return terms.sum()

    def norm_constant(self, n, m):
        """Normalization constant for the Zernike polynomials."""
        return np.sqrt((2 * (n + 1)) / (1 + (m == 0)))

    def evaluate(self, r, phi):
        """Compute the Zernike polynomial value at (r, phi)."""
        zernike_sum = 0
        for (n, m), coeff in zip(self.indices, self.coeffs):
            radial = self._radial_term(n, m, r)
            angular = np.cos(m * phi) if m >= 0 else np.sin(-m * phi)
            zernike_sum += coeff * self.norm_constant(n, m) * radial * angular
        return zernike_sum
