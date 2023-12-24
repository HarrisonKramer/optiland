import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


class ZernikeStandard:
    """OSA/ANSI Standard Zernike"""
    # TODO: add normalization constants

    def __init__(self, coeffs=[0 for _ in range(36)]):
        if len(coeffs) > 120:  # partial sum of first 15 natural numbers
            raise ValueError('Number of coefficients is limited to 120.')

        self.indices = self._generate_indices()
        self.coeffs = coeffs

    def get_term(self, coeff=0, n=0, m=0, r=0, phi=0):
        return (coeff *
                self._radial_term(n, m, r) *
                self._azimuthal_term(m, phi))

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
        indices_sorted = [element for _, element in
                          sorted(zip(number, indices))]

        # take only 120 indices
        return indices_sorted[:120]


class ZernikeFit:

    def __init__(self, x, y, z, zernike_type='fringe', num_terms=36):
        self.x = x
        self.y = y
        self.z = z
        self.type = zernike_type
        self.num_terms = num_terms

        self.radius = np.sqrt(self.x**2 + self.y**2)
        self.phi = np.arctan2(self.y, self.x)
        self.num_pts = np.size(self.z)

        if self.type == 'fringe':
            self.zernike = ZernikeFringe()
        elif self.type == 'standard':
            self.zernike = ZernikeStandard()
        else:
            raise ValueError('Zernike type must be "fringe" or "standard".')

        self._fit()

    @property
    def coeffs(self):
        return self.zernike.coeffs

    def view(self, projection='2d', num_points=128, figsize=(7, 5.5),
             z_label='OPD (waves)'):
        x, y = np.meshgrid(np.linspace(-1, 1, num_points),
                           np.linspace(-1, 1, num_points))
        radius = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        z = self.zernike.poly(radius, phi)

        z[radius > 1] = np.nan

        if projection == '2d':
            self._plot_2d(z, figsize=figsize, z_label=z_label)
        elif projection == '3d':
            self._plot_3d(x, y, z, figsize=figsize, z_label=z_label)
        else:
            raise ValueError('OPD projection must be "2d" or "3d".')

    def _plot_2d(self, z, figsize=(7, 5.5), z_label='OPD (waves)'):
        _, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(np.flipud(z), extent=[-1, 1, -1, 1])

        ax.set_xlabel('Pupil X')
        ax.set_ylabel('Pupil Y')
        ax.set_title(f'Zernike {self.type.capitalize()} Fit')

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(z_label, rotation=270)
        plt.show()

    def _plot_3d(self, x, y, z, figsize=(7, 5.5), z_label='OPD (waves)'):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                               figsize=figsize)

        surf = ax.plot_surface(x, y, z,
                               rstride=1, cstride=1,
                               cmap='viridis', linewidth=0,
                               antialiased=False)

        ax.set_xlabel('Pupil X')
        ax.set_ylabel('Pupil Y')
        ax.set_zlabel(z_label)
        ax.set_title(f'Zernike {self.type.capitalize()} Fit')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,
                     pad=0.15)
        fig.tight_layout()
        plt.show()

    def view_residual(self, figsize=(7, 5.5), z_label='Residual (waves)'):
        z = self.zernike.poly(self.radius, self.phi)
        rms = np.sqrt(np.mean((z-self.z)**2))

        _, ax = plt.subplots(figsize=figsize)
        s = ax.scatter(self.x, self.y, c=z-self.z)

        ax.set_xlabel('Pupil X')
        ax.set_ylabel('Pupil Y')
        ax.set_title(f'Residual: RMS={rms:.3}')

        cbar = plt.colorbar(s)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(z_label, rotation=270)
        plt.show()

    def _objective(self, coeffs):
        self.zernike.coeffs = coeffs
        z_computed = self.zernike.poly(self.radius, self.phi)
        return z_computed - self.z

    def _fit(self):
        initial_guess = [0 for _ in range(self.num_terms)]
        result = least_squares(self._objective, initial_guess)
        self.zernike.coeffs = result.x
