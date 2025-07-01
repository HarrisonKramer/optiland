"""Base PSF Module

This module provides a base class for Point Spread Function (PSF) calculations.

Kramer Harrison, 2025
"""

from abc import abstractmethod
from warnings import warn

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import zoom

import optiland.backend as be
from optiland.wavefront import Wavefront


def replace_nonpositive(image, min_value=1e-9):
    """
    Replace values <= 0 in the image with the smallest positive value in the image.
    If no positive value exists, use min_value.

    Args:
        image: Array (backend or numpy) to process.
        min_value: Value to use if no positive values exist (default: 1e-9).

    Returns:
        Array with non-positive values replaced.
    """
    if be.any(image <= 0):
        min_positive = be.min(image[image > 0]) if be.any(image > 0) else min_value
        return be.where(image <= 0, min_positive, image)
    return image


class BasePSF(Wavefront):
    """Base class for Point Spread Function (PSF) calculations.

    Args:
        optic (Optic): The optical system.
        field (tuple): The field as (x, y) at which to compute the PSF.
        wavelength (float): The wavelength of light.
        num_rays (int, optional): The number of rays used for wavefront
            computation. Defaults to 128.

    Attributes:
        psf (ndarray): The computed PSF. This should be set by subclasses.

    Methods:
        view(projection='2d', log=False, figsize=(7, 5.5), threshold=0.05,
            num_points=128): Visualizes the PSF.
    """

    def __init__(self, optic, field, wavelength, num_rays=128):
        super().__init__(
            optic=optic,
            fields=[field],
            wavelengths=[wavelength],
            num_rays=num_rays,
            distribution="uniform",
        )
        self.psf = None  # Subclasses must compute and set this

    def view(
        self,
        projection="2d",
        log=False,
        figsize=(7, 5.5),
        threshold=0.05,
        num_points=128,
    ):
        """Visualizes the PSF.

        Args:
            projection (str, optional): The projection type. Can be '2d' or
                '3d'. Defaults to '2d'.
            log (bool, optional): Whether to use a logarithmic scale for the
                intensity. Defaults to False.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).
            threshold (float, optional): The threshold for determining the
                bounds of the PSF for zoomed view. Defaults to 0.05.
            num_points (int, optional): The number of points used for
                interpolating the PSF for smoother visualization. Defaults to 128.

        Raises:
            ValueError: If the projection is not '2d' or '3d'.
            RuntimeError: If the PSF has not been computed by the subclass.
        """
        if self.psf is None:
            raise RuntimeError(
                "PSF has not been computed. Call _compute_psf in subclass."
            )

        psf_np = be.to_numpy(self.psf)
        min_x, min_y, max_x, max_y = self._find_bounds(psf_np, threshold)
        psf_zoomed = psf_np[min_x:max_x, min_y:max_y]

        oversampling_factor = num_points / psf_zoomed.shape[0]

        if oversampling_factor > 3:
            message = (
                f"The PSF view has a high oversampling factor "
                f"({oversampling_factor:.2f}). Results may be inaccurate."
            )
            warn(message, stacklevel=2)

        # Subclasses should implement _get_psf_units if they want physical units
        # otherwise, pixel units are used.
        if hasattr(self, "_get_psf_units"):
            x_extent, y_extent = self._get_psf_units(psf_zoomed)
            x_label, y_label = "X (µm)", "Y (µm)"
        else:
            # Default to pixel units if not implemented by subclass
            x_extent = psf_zoomed.shape[1]
            y_extent = psf_zoomed.shape[0]
            x_label, y_label = "X (pixels)", "Y (pixels)"

        psf_smooth = self._interpolate_psf(psf_zoomed, num_points)

        if projection == "2d":
            self._plot_2d(
                psf_smooth,
                log,
                x_extent,
                y_extent,
                figsize,
                x_label,
                y_label,
                psf_zoomed.shape,
            )
        elif projection == "3d":
            self._plot_3d(
                psf_smooth,
                log,
                x_extent,
                y_extent,
                figsize,
                x_label,
                y_label,
                psf_zoomed.shape,
            )
        else:
            raise ValueError('Projection must be "2d" or "3d".')

    def _plot_2d(
        self, image, log, x_extent, y_extent, figsize, x_label, y_label, original_size
    ):
        """Plots the PSF in 2D.

        Args:
            image (numpy.ndarray): The 2D image of the PSF to plot.
            log (bool): If True, apply logarithmic normalization to the image.
            x_extent (float): The extent of the x-axis.
            y_extent (float): The extent of the y-axis.
            figsize (tuple): The size of the figure.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            original_size (tuple): The original size of the PSF image before
                interpolation.
        """
        fig, ax = plt.subplots(figsize=figsize)
        norm = LogNorm() if log else None

        # Replace values <= 0 with smallest non-zero value in image for log scale
        if log and be.any(image <= 0):
            image = replace_nonpositive(image)

        extent = [-x_extent / 2, x_extent / 2, -y_extent / 2, y_extent / 2]
        im = ax.imshow(be.to_numpy(image), norm=norm, extent=extent)

        self._annotate_original_size(fig, original_size)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{self.__class__.__name__.replace('PSF', ' PSF')}")

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Relative Intensity (%)", rotation=270)
        plt.show()

    def _plot_3d(
        self, image, log, x_extent, y_extent, figsize, x_label, y_label, original_size
    ):
        """Plots the PSF in 3D.

        Args:
            image (numpy.ndarray): The PSF image data.
            log (bool): Whether to apply logarithmic scaling to the image.
            x_extent (float): The extent of the x-axis.
            y_extent (float): The extent of the y-axis.
            figsize (tuple): The size of the figure.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            original_size (tuple): The original size of the PSF image before
                interpolation.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)

        x_np = be.to_numpy(be.linspace(-x_extent / 2, x_extent / 2, image.shape[1]))
        y_np = be.to_numpy(be.linspace(-y_extent / 2, y_extent / 2, image.shape[0]))
        X_np, Y_np = np.meshgrid(x_np, y_np)

        # Replace values <= 0 with smallest non-zero value in image for log scale
        if log and be.any(image <= 0):
            image = replace_nonpositive(image)

        image_np = be.to_numpy(image)

        log_formatter = None
        if log:
            image_plot = np.log10(image_np)
            formatter = mticker.FuncFormatter(self._log_tick_formatter)
            ax.zaxis.set_major_formatter(formatter)
            ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            log_formatter = self._log_colorbar_formatter
        else:
            image_plot = image_np

        surf = ax.plot_surface(
            X_np,
            Y_np,
            image_plot,
            rstride=1,
            cstride=1,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
        )

        self._annotate_original_size(fig, original_size)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel("Relative Intensity (%)")
        ax.set_title(f"{self.__class__.__name__.replace('PSF', ' PSF')}")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15, format=log_formatter)
        fig.tight_layout()
        plt.show()

    def _log_tick_formatter(self, value, pos=None):
        """Formats tick labels for a logarithmic scale (Z-axis in 3D plot)."""
        return f"$10^{{{int(value)}}}$"

    def _log_colorbar_formatter(self, value, pos=None):
        """Formats tick labels for a logarithmic colorbar."""
        linear_value = 10**value
        return f"{linear_value:.1e}"

    def _annotate_original_size(self, fig: plt.Figure, original_size):
        """Annotates the original size of the zoomed PSF in the bottom right corner."""
        text = f"Original Size: {original_size[0]}×{original_size[1]}"
        fig.text(
            0.99,
            0.01,
            text,
            transform=fig.transFigure,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    def _interpolate_psf(self, image, n=128):
        """Interpolates the PSF for visualization.

        Uses scipy.ndimage.zoom for interpolation. Converts to NumPy
        as zoom requires NumPy array.

        Args:
            image (ndarray): The input image (can be backend array).
            n (int, optional): The number of points in the interpolated
                grid. Defaults to 128.

        Returns:
            ndarray: The interpolated PSF grid (backend array).
        """
        image_np = be.to_numpy(image)
        zoom_factor = n / image_np.shape[0]

        if zoom_factor == 1:
            return image  # Return original backend array if no zoom

        interpolated_np = zoom(image_np, zoom_factor, order=3)
        return be.array(interpolated_np)

    @staticmethod
    def _find_bounds(psf, threshold=0.25):
        """Finds the bounding box coordinates for the non-zero elements in the
        PSF matrix.

        Args:
            psf (numpy.ndarray): The PSF matrix.
            threshold (float): The threshold value for determining non-zero
                elements in the PSF matrix. Default is 0.25.

        Returns:
            tuple: A tuple containing the minimum and maximum x and y
                coordinates of the bounding box.

        """
        thresholded_psf = psf > threshold
        non_zero_indices = np.argwhere(thresholded_psf)

        try:
            min_x, min_y = np.min(non_zero_indices, axis=0)
            max_x, max_y = np.max(non_zero_indices, axis=0)
        except ValueError:
            min_x, min_y = 0, 0
            max_x, max_y = psf.shape

        size = max(max_x - min_x, max_y - min_y)

        peak_x, peak_y = psf.shape[0] // 2, psf.shape[1] // 2

        min_x = peak_x - size / 2
        max_x = peak_x + size / 2
        min_y = peak_y - size / 2
        max_y = peak_y + size / 2

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(psf.shape[0], max_x)
        max_y = min(psf.shape[1], max_y)

        return int(min_x), int(min_y), int(max_x), int(max_y)

    @abstractmethod
    def _compute_psf(self):
        """Computes the PSF.

        This method must be implemented by subclasses.
        It should calculate the PSF and store it in self.psf.
        """
        raise NotImplementedError("Subclasses must implement _compute_psf.")

    def strehl_ratio(self):
        """Computes the Strehl ratio of the PSF.

        The Strehl ratio is the ratio of the peak intensity of the aberrated
        PSF to the peak intensity of the diffraction-limited PSF.
        Assumes self.psf is normalized such that its peak would be 1.0 (or 100%)
        for a diffraction-limited system.

        Returns:
            float: The Strehl ratio.

        Raises:
            RuntimeError: If the PSF has not been computed.
        """
        if self.psf is None:
            raise RuntimeError("PSF has not been computed.")

        center_x = self.psf.shape[0] // 2
        center_y = self.psf.shape[1] // 2
        return self.psf[center_x, center_y] / 100

    def _get_effective_FNO(self):
        """Calculates the effective F-number of the optical system.

        Returns:
            float: The effective F-number.
        """
        FNO = self.optic.paraxial.FNO()

        if not self.optic.object_surface.is_infinite:
            D = self.optic.paraxial.XPD()
            p = D / self.optic.paraxial.EPD()
            m = self.optic.paraxial.magnification()
            FNO = FNO * (1 + be.abs(m) / p)

        return FNO
