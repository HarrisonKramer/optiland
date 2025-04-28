"""Zernike Fit Module

This module contains the ZernikeFit class, which can be used to fit Zernike
polynomial to a set of points. This is commonly used for wavefront calculations,
but the class can be used for any fitting operation requiring Zernike polynomials.

Kramer Harrison, 2025
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optiland import backend as be
from optiland.zernike import ZernikeFringe, ZernikeNoll, ZernikeStandard

ZERNIKE_CLASSES = {
    "fringe": ZernikeFringe,
    "standard": ZernikeStandard,
    "noll": ZernikeNoll,
}


class ZernikeFit:
    """
    Fit Zernike polynomials to wavefront or arbitrary data points.

    This class constructs a linear design matrix of Zernike basis functions and solves
    for the coefficients via least squares.

    Args:
        x (array-like): X-coordinates of data points.
        y (array-like): Y-coordinates of data points.
        z (array-like): Values at (x, y) to fit.
        zernike_type (str): Type of Zernike basis: 'fringe', 'standard', or 'noll'.
        num_terms (int): Number of Zernike terms to include in the fit.

    Attributes:
        x (array-like): Flattened x-coordinates.
        y (array-like): Flattened y-coordinates.
        z (array-like): Flattened target values.
        radius (array-like): Radial coordinate of each point.
        phi (array-like): Azimuthal coordinate of each point.
        num_pts (int): Number of data points.
        zernike (BaseZernike): Zernike basis instance with fitted coefficients.
    """

    def __init__(
        self,
        x,
        y,
        z,
        zernike_type: str = "fringe",
        num_terms: int = 36,
    ):
        # Convert inputs to backend tensors and flatten
        self.x = be.asarray(x).reshape(-1)
        self.y = be.asarray(y).reshape(-1)
        self.z = be.asarray(z).reshape(-1)

        if self.x.shape != self.y.shape or self.x.shape != self.z.shape:
            raise ValueError("`x`, `y`, and `z` must have the same number of elements.")

        self.num_terms = num_terms
        self.num_pts = int(be.size(self.x))

        # Compute polar coordinates
        self.radius = be.sqrt(self.x**2 + self.y**2)
        self.phi = be.arctan2(self.y, self.x)

        # Validate Zernike type and instantiate basis
        if zernike_type not in ZERNIKE_CLASSES:
            raise ValueError(
                f"Invalid Zernike type '{zernike_type}'. "
                f"Choose from: {list(ZERNIKE_CLASSES)}"
            )
        self.zernike_type = zernike_type
        self.zernike = ZERNIKE_CLASSES[zernike_type](be.ones(num_terms))

        # Fit coefficients
        self._fit()

    @property
    def coeffs(self):
        """
        Tensor: The fitted Zernike coefficients.
        """
        return self.zernike.coeffs

    def _fit(self):
        """
        Build design matrix of Zernike basis functions and solve linear least squares.
        """
        # Build design matrix A
        A = be.stack(self.zernike.terms(self.radius, self.phi), axis=1)

        # Solve linear least squares A c = z
        try:
            solution = be.linalg.lstsq(A, self.z, rcond=None)
            coeffs = solution[0]
        except (AttributeError, TypeError):
            # Fallback via pseudoinverse
            pinv = be.linalg.pinv(A)
            coeffs = be.matmul(pinv, self.z)

        # Assign coefficients to the main zernike instance
        self.zernike.coeffs = coeffs

    def view(
        self,
        projection: str = "2d",
        num_points: int = 128,
        figsize: tuple[float, float] = (7, 5.5),
        z_label: str = "OPD (waves)",
    ):
        """
        Visualize the fitted Zernike surface.

        Args:
            projection (str): '2d' for image plot, '3d' for surface plot.
            num_points (int): Grid resolution for display.
            figsize (tuple): Figure size in inches.
            z_label (str): Label for the z-axis or colorbar.

        Raises:
            ValueError: If `projection` is not '2d' or '3d'.
        """
        # Create grid in unit circle
        grid_x, grid_y = be.meshgrid(
            be.linspace(-1.0, 1.0, num_points),
            be.linspace(-1.0, 1.0, num_points),
        )
        grid_r = be.sqrt(grid_x**2 + grid_y**2)
        grid_phi = be.arctan2(grid_y, grid_x)
        grid_z = self.zernike.poly(grid_r, grid_phi)

        # Mask outside unit circle
        grid_z = be.where(grid_r > 1.0, be.nan, grid_z)

        # Convert to NumPy for plotting
        x_np = be.to_numpy(grid_x)
        y_np = be.to_numpy(grid_y)
        z_np = be.to_numpy(grid_z)

        if projection == "2d":
            self._plot_2d(z_np, figsize=figsize, z_label=z_label)
        elif projection == "3d":
            self._plot_3d(x_np, y_np, z_np, figsize=figsize, z_label=z_label)
        else:
            raise ValueError("`projection` must be '2d' or '3d'.")

    def view_residual(
        self,
        figsize: tuple[float, float] = (7, 5.5),
        z_label: str = "Residual (waves)",
    ):
        """
        Scatter plot of residuals between fitted surface and original data.

        Args:
            figsize (tuple): Figure size in inches.
            z_label (str): Label for the colorbar.
        """
        # Compute fitted values and residuals
        fitted = self.zernike.poly(self.radius, self.phi)
        residuals = fitted - self.z
        rms = be.sqrt(be.mean(residuals**2))

        _, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(
            be.to_numpy(self.x),
            be.to_numpy(self.y),
            c=be.to_numpy(residuals),
            marker="o",
            edgecolors="none",
        )
        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_title(f"Residuals (RMS={rms:.3f})")
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(z_label, rotation=270, labelpad=15)
        plt.show()

    def _plot_2d(self, z, figsize, z_label):
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
        ax.set_title(f"Zernike {self.zernike_type.capitalize()} Fit")
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel(z_label, rotation=270, labelpad=15)
        plt.show()

    def _plot_3d(self, x, y, z, figsize, z_label):
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
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
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
        ax.set_title(f"Zernike {self.zernike_type.capitalize()} Fit")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15)
        plt.tight_layout()
        plt.show()
