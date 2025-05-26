"""Base PSF Module

This module provides a base class for Point Spread Function (PSF) calculations.

Kramer Harrison, 2023 (Originally in fft.py)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import zoom

import optiland.backend as be
from optiland.wavefront import Wavefront


class BasePSF(Wavefront):
    """Base class for Point Spread Function (PSF) calculations.

    Args:
        optic (Optic): The optical system.
        field (tuple): The field as (x, y) at which to compute the PSF.
        wavelength (float): The wavelength of light.
        num_rays (int, optional): The number of rays used for wavefront
            computation. Defaults to 128.
        grid_size (int, optional): The size of the grid used for internal
            PSF computations. Defaults to 1024. Not all subclasses may use
            this directly.

    Attributes:
        grid_size (int): The size of the grid.
        psf (ndarray): The computed PSF. This should be set by subclasses.

    Methods:
        view(projection='2d', log=False, figsize=(7, 5.5), threshold=0.05,
            num_points=128): Visualizes the PSF.
    """

    def __init__(self, optic, field, wavelength, num_rays=128, grid_size=1024):
        super().__init__(
            optic=optic,
            fields=[field],  # PSF is usually for a single field point
            wavelengths=[wavelength],  # And single wavelength
            num_rays=num_rays,
            distribution="uniform",  # Uniform distribution for pupil sampling
        )
        self.grid_size = grid_size
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
            self._plot_2d(psf_smooth, log, x_extent, y_extent, figsize, x_label, y_label)
        elif projection == "3d":
            self._plot_3d(psf_smooth, log, x_extent, y_extent, figsize, x_label, y_label)
        else:
            raise ValueError('Projection must be "2d" or "3d".')

    def _plot_2d(self, image, log, x_extent, y_extent, figsize, x_label, y_label):
        """Plots the PSF in 2D.

        Args:
            image (numpy.ndarray): The 2D image of the PSF to plot.
            log (bool): If True, apply logarithmic normalization to the image.
            x_extent (float): The extent of the x-axis.
            y_extent (float): The extent of the y-axis.
            figsize (tuple): The size of the figure.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
        """
        _, ax = plt.subplots(figsize=figsize)
        norm = LogNorm() if log else None

        # Replace values <= 0 with smallest non-zero value in image for log scale
        if log and be.any(image <= 0):
            min_positive = be.min(image[image > 0]) if be.any(image > 0) else 1e-9
            image = be.where(image <= 0, min_positive, image)


        extent = [-x_extent / 2, x_extent / 2, -y_extent / 2, y_extent / 2]
        im = ax.imshow(be.to_numpy(image), norm=norm, extent=extent, origin='lower')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{self.__class__.__name__}")

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Relative Intensity (%)", rotation=270)
        plt.show()

    def _plot_3d(self, image, log, x_extent, y_extent, figsize, x_label, y_label):
        """Plots the PSF in 3D.

        Args:
            image (numpy.ndarray): The PSF image data.
            log (bool): Whether to apply logarithmic scaling to the image.
            x_extent (float): The extent of the x-axis.
            y_extent (float): The extent of the y-axis.
            figsize (tuple): The size of the figure.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)

        x_np = be.to_numpy(be.linspace(-x_extent / 2, x_extent / 2, image.shape[1]))
        y_np = be.to_numpy(be.linspace(-y_extent / 2, y_extent / 2, image.shape[0]))
        X_np, Y_np = np.meshgrid(x_np, y_np) # meshgrid is fine with numpy for plotting

        image_np = be.to_numpy(image)

        # Replace values <= 0 with smallest non-zero value in image for log scale
        if log and np.any(image_np <= 0):
            min_positive = np.min(image_np[image_np > 0]) if np.any(image_np > 0) else 1e-9
            image_np[image_np <= 0] = min_positive


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

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel("Relative Intensity (%)")
        ax.set_title(f"{self.__class__.__name__}")

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
            return image # Return original backend array if no zoom
        
        interpolated_np = zoom(image_np, zoom_factor, order=3)
        return be.array(interpolated_np)


    @staticmethod
    def _find_bounds(psf_np, threshold=0.25):
        """Finds bounding box for non-zero elements in the PSF matrix.

        Operates on NumPy arrays as it uses np.argwhere and np.min/max.

        Args:
            psf_np (numpy.ndarray): The PSF matrix (NumPy array).
            threshold (float): Threshold for determining non-zero elements.
                Defaults to 0.25.

        Returns:
            tuple: Min/max x and y coordinates of the bounding box.
        """
        if not isinstance(psf_np, np.ndarray):
            raise TypeError("_find_bounds expects a NumPy array.")

        thresholded_psf = psf_np > threshold * be.max(psf_np) # Threshold relative to max
        non_zero_indices = np.argwhere(thresholded_psf)

        if non_zero_indices.size == 0: # Handle case with no points above threshold
            min_x, min_y = 0, 0
            max_x, max_y = psf_np.shape[0], psf_np.shape[1]
            return int(min_x), int(min_y), int(max_x), int(max_y)

        min_x_idx, min_y_idx = np.min(non_zero_indices, axis=0)
        max_x_idx, max_y_idx = np.max(non_zero_indices, axis=0)
        
        # Ensure a certain size around the peak if the thresholded area is too small
        # This helps to get a consistent view centered around the PSF core
        size = max(max_x_idx - min_x_idx, max_y_idx - min_y_idx, psf_np.shape[0] // 10) # Ensure min size

        # Center the bounds around the actual peak of the PSF, not the center of thresholded area
        peak_coord = np.unravel_index(np.argmax(psf_np), psf_np.shape)
        
        center_x, center_y = peak_coord[0], peak_coord[1]

        min_x = center_x - size / 2
        max_x = center_x + size / 2
        min_y = center_y - size / 2
        max_y = center_y + size / 2

        # Clip to array bounds
        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(psf_np.shape[0], int(max_x))
        max_y = min(psf_np.shape[1], int(max_y))
        
        # Ensure max_x > min_x and max_y > min_y
        if max_x <= min_x: max_x = min_x + 1
        if max_y <= min_y: max_y = min_y + 1
        max_x = min(psf_np.shape[0], max_x) # re-clip
        max_y = min(psf_np.shape[1], max_y) # re-clip


        return min_x, min_y, max_x, max_y

    # Abstract method placeholder for subclasses to implement
    def _compute_psf(self):
        """Computes the PSF.
        
        This method must be implemented by subclasses.
        It should calculate the PSF and store it in self.psf.
        """
        raise NotImplementedError("Subclasses must implement _compute_psf.")

    # Placeholder for strehl ratio, can be overridden by subclasses if specific
    # normalization or calculation is needed.
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
            NotImplementedError: If the subclass does not define how grid_size
                                 relates to the PSF's center peak.
        """
        if self.psf is None:
            raise RuntimeError("PSF has not been computed.")
        
        # This default implementation assumes a grid where the peak is at grid_size // 2
        # Subclasses like FFTPSF use this. HuygensPSF might need to override if its
        # PSF grid center is different or if normalization is handled differently.
        if not hasattr(self, 'grid_size') or self.grid_size is None:
             # Find peak dynamically if grid_size is not a reliable indicator
            peak_intensity = be.max(self.psf)
            # This assumes the PSF is already normalized to a diffraction-limited peak of 100
            # This might not be true for all PSF computation methods.
            # Consider requiring subclasses to provide their diffraction-limited peak.
            return peak_intensity / 100.0 # Or handle normalization within _compute_psf

        # Default for FFTPSF-like grids
        center_idx = self.grid_size // 2
        
        # Ensure PSF is 2D and large enough
        if len(self.psf.shape) < 2 or \
           center_idx >= self.psf.shape[0] or \
           center_idx >= self.psf.shape[1]:
            # Fallback to max if center index is out of bounds (e.g. HuygensPSF)
            # This still assumes normalization to 100 for diffraction-limited peak.
             peak_intensity = be.max(self.psf)
             return peak_intensity / 100.0

        return self.psf[center_idx, center_idx] / 100.0
