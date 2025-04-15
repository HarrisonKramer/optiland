"""Zernike Fit Module

This module contains the ZernikeFit class, which can be used to fit Zernike
polynomial to a set of points. This is commonly used for wavefront calculations,
but the class can be used for any fitting operation requiring Zernike polynomials.

Kramer Harrison, 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from optiland.zernike import ZernikeFringe, ZernikeNoll, ZernikeStandard

ZERNIKE_CLASSES = {
    "fringe": ZernikeFringe,
    "standard": ZernikeStandard,
    "noll": ZernikeNoll,
}


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

        if self.type not in ZERNIKE_CLASSES:
            raise ValueError(
                f"Invalid Zernike type '{self.type}'. "
                f"Valid types are: {list(ZERNIKE_CLASSES.keys())}"
            )
        self.zernike = ZERNIKE_CLASSES[self.type]()

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
