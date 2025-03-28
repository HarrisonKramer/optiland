"""Zernike Module

This module provides functionality for working with Zernike polynomials, which
are used to represent wavefront aberrations. The `ZernikeStandard` class
implements the OSA/ANSI standard Zernike polynomials, allowing for the
calculation of Zernike terms and the evaluation of Zernike polynomial series
for given radial and azimuthal coordinates. The module also provides classes
for Zernike Fringe (University of Arizona) and Zernike Noll indices. Lastly,
a `ZernikeFit` class is provided for fitting the various Zernike polynomial
types to data points.

Kramer Harrison, 2023
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


class ZernikeStandard:
    """OSA/ANSI Standard Zernike

    This class represents the OSA/ANSI Standard Zernike polynomials.
    It provides methods to calculate the Zernike terms, Zernike polynomials,
    and other related functions.

    Args:
        coeffs (list): the coefficient list for the Zernike polynomials.
            Defaults to all zeros (36 elements total)

    References:
        1. https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/
           ANSI_standard_indices
        2. Thibos LN, Applegate RA, Schwiegerling JT, Webb R; VSIA Standards
           Taskforce Members. Vision science and its applications. Standards
           for reporting the optical aberrations of eyes. J Refract Surg. 2002
           Sep-Oct;18(5):S652-60. doi: 10.3928/1081-597X-20020901-30. PMID:
           12361175.

    """

    def __init__(self, coeffs=None):
        if coeffs is None:
            coeffs = [0 for _ in range(36)]

        if len(coeffs) > 120:  # partial sum of first 15 natural numbers
            raise ValueError("Number of coefficients is limited to 120.")

        self.indices = self._generate_indices()
        self.coeffs = coeffs

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

    def _radial_term(self, n=0, m=0, r=0):
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

    def _norm_constant(self, n=0, m=0):
        """Calculate the normalization constant of the Zernike polynomial.

        Args:
            n (int): Radial order of the Zernike term.
            m (int): Azimuthal order of the Zernike term.

        Returns:
            float: The calculated value of the normalization constant.

        """
        return np.sqrt((2 * n + 2) / (1 + (m == 0)))

    def _generate_indices(self):
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


class ZernikeFringe(ZernikeStandard):
    """Zernike Fringe Coefficients

    This class represents Zernike Fringe Coefficients. It is a subclass of the
    ZernikeStandard class.

    Args:
        coeffs (list): the coefficient list for the Zernike polynomials.
            Defaults to all zeros (36 elements total)

    References:
        1. https://en.wikipedia.org/wiki/Zernike_polynomials#Fringe/
           University_of_Arizona_indices

    """

    def __init__(self, coeffs=None):
        if coeffs is None:
            coeffs = [0 for _ in range(36)]
        super().__init__(coeffs)

    def _norm_constant(self, n=0, m=0):
        """Calculate the normalization constant for a given Zernike polynomial.
        Note that this is 1 for all terms for Zernike Fringe polynomials.

        Args:
            n (int): The radial order of the Zernike polynomial.
            m (int): The azimuthal order of the Zernike polynomial.

        Returns:
            float: The normalization constant for the Zernike polynomial.

        """
        return 1

    def _generate_indices(self):
        """Generate the indices for the Zernike terms.

        Returns:
            list: List of tuples representing the indices (n, m) of the
                Zernike terms.

        """
        number = []
        indices = []
        for n in range(20):
            for m in range(-n, n + 1):
                if (n - m) % 2 == 0:
                    number.append(
                        int(
                            (1 + (n + np.abs(m)) / 2) ** 2
                            - 2 * np.abs(m)
                            + (1 - np.sign(m)) / 2,
                        ),
                    )
                    indices.append((n, m))

        # sort indices according to fringe coefficient number
        indices_sorted = [element for _, element in sorted(zip(number, indices))]

        # take only 120 indices
        return indices_sorted[:120]


class ZernikeNoll(ZernikeStandard):
    """Zernike Coefficients - Noll Standard

    This class represents Zernike Noll Coefficients. It is a subclass of the
    ZernikeStandard class. Note that the Noll notation is used for the
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

    def _norm_constant(self, n=0, m=0):
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

    def _generate_indices(self):
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


class ZernikeFit:
    """Class for fitting Zernike polynomials to wavefront data.

    Args:
        x (array-like): x-coordinates of the wavefront data.
        y (array-like): y-coordinates of the wavefront data.
        z (array-like): z-coordinates (wavefront) of the data.
        zernike_type (str, optional): Type of Zernike polynomials to use.
            Default is 'fringe'.
        num_terms (int, optional): Number of Zernike terms to fit.
            Default is 36.

    Attributes:
        x (array-like): x-coordinates of the wavefront data.
        y (array-like): y-coordinates of the wavefront data.
        z (array-like): z-coordinates (wavefront) of the data.
        type (str): Type of Zernike polynomials used.
        num_terms (int): Number of Zernike terms used.
        radius (array-like): Distance from the origin for each point in the
            wavefront data.
        phi (array-like): Angle from the x-axis for each point in the
            wavefront data.
        num_pts (int): Number of points in the wavefront data.
        zernike (ZernikeBase): Zernike polynomial object used for fitting.
        coeffs (array-like): Coefficients of the fitted Zernike polynomials.

    Methods:
        view(projection='2d', num_points=128, figsize=(7, 5.5),
            z_label='OPD (waves)'): Visualize the fitted Zernike polynomials.
        view_residual(figsize=(7, 5.5), z_label='Residual (waves)'): Visualize
            the residual between the fitted Zernike polynomials and the
            original data.

    """

    def __init__(self, x, y, z, zernike_type="fringe", num_terms=36):
        self.x = x
        self.y = y
        self.z = z
        self.type = zernike_type
        self.num_terms = num_terms

        self.radius = np.sqrt(self.x**2 + self.y**2)
        self.phi = np.arctan2(self.y, self.x)
        self.num_pts = np.size(self.z)

        if self.type == "fringe":
            self.zernike = ZernikeFringe()
        elif self.type == "standard":
            self.zernike = ZernikeStandard()
        elif self.type == "noll":
            self.zernike = ZernikeNoll()
        else:
            raise ValueError('Zernike type must be "fringe", "standard", or "noll".')

        self._fit()

    @property
    def coeffs(self):
        """list: coefficients of the Zernike fit"""
        return self.zernike.coeffs

    def view(
        self,
        projection="2d",
        num_points=128,
        figsize=(7, 5.5),
        z_label="OPD (waves)",
    ):
        """Visualizes the Zernike polynomial representation of the wavefront.

        Args:
            projection (str): The type of projection to use for visualization.
                Can be '2d' or '3d'.
            num_points (int): The number of points to sample along each axis
                for the visualization.
            figsize (tuple): The size of the figure in inches.
                Defaults to (7, 5.5).
            z_label (str): The label for the z-axis in the visualization.
                Defaults to 'OPD (waves)'.

        Raises:
            ValueError: If the projection is not '2d' or '3d'.

        """
        x, y = np.meshgrid(
            np.linspace(-1, 1, num_points),
            np.linspace(-1, 1, num_points),
        )
        radius = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        z = self.zernike.poly(radius, phi)

        z[radius > 1] = np.nan

        if projection == "2d":
            self._plot_2d(z, figsize=figsize, z_label=z_label)
        elif projection == "3d":
            self._plot_3d(x, y, z, figsize=figsize, z_label=z_label)
        else:
            raise ValueError('OPD projection must be "2d" or "3d".')

    def _plot_2d(self, z, figsize=(7, 5.5), z_label="OPD (waves)"):
        """Plot a 2D representation of the given data.

        Args:
            z (numpy.ndarray): The data to be plotted.
            figsize (tuple, optional): The size of the figure
                (default is (7, 5.5)).
            z_label (str, optional): The label for the colorbar
                (default is 'OPD (waves)').

        """
        _, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(np.flipud(z), extent=[-1, 1, -1, 1])

        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_title(f"Zernike {self.type.capitalize()} Fit")

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(z_label, rotation=270)
        plt.show()

    def _plot_3d(self, x, y, z, figsize=(7, 5.5), z_label="OPD (waves)"):
        """Plot a 3D surface plot of the given data.

        Args:
            x (numpy.ndarray): Array of x-coordinates.
            y (numpy.ndarray): Array of y-coordinates.
            z (numpy.ndarray): Array of z-coordinates.
            figsize (tuple, optional): Size of the figure (width, height).
                Default is (7, 5.5).
            z_label (str, optional): Label for the z-axis.
                Default is 'OPD (waves)'.

        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)

        surf = ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
        )

        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_zlabel(z_label)
        ax.set_title(f"Zernike {self.type.capitalize()} Fit")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15)
        fig.tight_layout()
        plt.show()

    def view_residual(self, figsize=(7, 5.5), z_label="Residual (waves)"):
        """Visualizes the residual of the Zernike polynomial fit.

        Args:
            figsize (tuple): The size of the figure (width, height).
                Default is (7, 5.5).
            z_label (str): The label for the colorbar representing the
                residual. Default is 'Residual (waves)'.

        """
        z = self.zernike.poly(self.radius, self.phi)
        rms = np.sqrt(np.mean((z - self.z) ** 2))

        _, ax = plt.subplots(figsize=figsize)
        s = ax.scatter(self.x, self.y, c=z - self.z)

        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_title(f"Residual: RMS={rms:.3}")

        cbar = plt.colorbar(s)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(z_label, rotation=270)
        plt.show()

    def _objective(self, coeffs):
        """Compute the objective value for the optimization problem.

        Args:
            coeffs (array-like): Coefficients for the Zernike polynomial.

        Returns:
            float: The computed objective value.

        """
        self.zernike.coeffs = coeffs
        z_computed = self.zernike.poly(self.radius, self.phi)
        return z_computed - self.z

    def _fit(self):
        """Fits the Zernike coefficients by minimizing the objective function."""
        initial_guess = [0 for _ in range(self.num_terms)]
        result = least_squares(self._objective, initial_guess)
        self.zernike.coeffs = result.x
