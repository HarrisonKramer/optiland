import math
import numpy as np


class ZernikeStandard:
    """OSA/ANSI Standard Zernike"""

    def __init__(self, coeffs=[0 for _ in range(36)]):
        if len(coeffs) > 120:  # partial sum of first 15 natural numbers
            raise ValueError('Number of coefficients is limited to 120.')

        self.indices = self._generate_indices()
        self.coeffs = coeffs

    def get_term(self, coeff=0, n=0, m=0, r=0, phi=0):
        return coeff * self._radial_term(n, m, r) * self._azimuthal_term(m, phi)

    def terms(self, r=0, phi=0):
        val = []
        for k, idx in enumerate(self.indices):
            n, m = idx
            try:
                val.append(self.get_term(self.coeffs[k], n, m, r, phi))
            except IndexError:
                break
        return val

    def poly(self, r=0, phi=0):
        return sum(self.terms(r, phi))

    def _radial_term(self, n=0, m=0, r=0):
        s_max = int((n - np.abs(m)) / 2 + 1)
        value = 0
        for k in range(s_max):
            value += (-1)**k * math.factorial(n - k) / \
                (math.factorial(k) *
                 math.factorial(int((n + m)/2 - k)) *
                 math.factorial(int((n - m)/2 - k))) * r ** (n - 2*k)
        return value

    def _azimuthal_term(self, m=0, phi=0):
        if m >= 0:
            return np.cos(m * phi)
        else:
            return np.sin(m * phi)

    def _generate_indices(self):
        indices = []
        for n in range(15):
            for m in range(-n, n+1):
                if (n - m) % 2 == 0:
                    indices.append((n, m))
        return indices


class ZernikeFringe(ZernikeStandard):
    """Zernike Fringe Coefficients"""

    def __init__(self, terms=[0 for _ in range(36)]):
        super().__init__(terms)

    def _generate_indices(self):
        number = []
        indices = []
        for n in range(20):
            for m in range(-n, n+1):
                if (n - m) % 2 == 0:
                    number.append(int((1 + (n + np.abs(m))/2)**2 -
                                      2 * np.abs(m) + (1 - np.sign(m)) / 2))
                    indices.append((n, m))

        # sort indices according to fringe coefficient number
        indices_sorted = [element for _, element in sorted(zip(number, indices))]

        # take only 120 indices
        return indices_sorted[:120]


class ZernikeFit:

    def __init__(self):
        pass
