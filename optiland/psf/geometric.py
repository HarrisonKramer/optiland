"""Geometric Point Spread Function (PSF) Module

This module provides functionality for simulating the Point Spread Function (PSF)
of optical systems based on geometric ray tracing (spot diagrams).

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.analysis import SpotDiagram
from optiland.psf.base import BasePSF


class GeometricPSF(BasePSF):
    """Class representing the Geometric PSF.

    This class computes the PSF of an optical system by generating a 2D histogram
    of ray intersection points with the image plane. It inherits common
    visualization and initialization functionalities from `BasePSF`.

    The geometric PSF is essentially a high-resolution spot diagram.

    Args:
        optic (Optic): The optical system object.
        field (tuple): The field point (e.g., (Hx, Hy) in normalized field
            coordinates) at which to compute the PSF.
        wavelength (float): The wavelength of light in micrometers.
        num_rays (int, optional): The number of rays to trace for generating
            the spot diagram. Defaults to 10000.
        grid_size (int, optional): The number of bins on each side of the 2D
            histogram used to represent the PSF. Defaults to 256.
        distribution (str, optional): The distribution of rays for SpotDiagram.
            Defaults to 'uniform'.

    Attributes:
        psf (be.ndarray): The computed Geometric PSF. This is a 2D array
            representing the density of ray hits, normalized to a peak of 100.
        grid_size (int): The number of bins used for the histogram.
        spot_data (SpotDiagram): The underlying spot diagram data.
        x_edges (be.ndarray): The bin edges for the x-axis of the histogram.
        y_edges (be.ndarray): The bin edges for the y-axis of the histogram.
    """

    def __init__(
        self,
        optic,
        field,
        wavelength,
        num_rays=256,
        grid_size=256,
        distribution="uniform",
    ):
        super().__init__(
            optic=optic, field=field, wavelength=wavelength, num_rays=num_rays
        )
        self.grid_size = grid_size

        self.spot_data = SpotDiagram(
            optic=optic,
            fields=[field],  # SpotDiagram expects a list of fields
            wavelengths=[wavelength],  # and a list of wavelengths
            num_rays=num_rays,
            distribution=distribution,
        )
        # SpotDiagram stores data in a list of lists (fields, then wavelengths)
        # For GeometricPSF, we have one field and one wavelength.
        self.ray_intersections_x = self.spot_data.data[0][0].x
        self.ray_intersections_y = self.spot_data.data[0][0].y
        self.ray_intensities = self.spot_data.data[0][0].intensity

        self.psf, self.x_edges, self.y_edges = self._compute_psf()

    def _compute_psf(self):
        """Computes the Geometric PSF from ray intersection data.

        This involves:
        1. Creating a 2D histogram of the ray (x, y) intersection points
           on the image plane.
        2. Normalizing the histogram so its peak is 100.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]:
                - psf_image (be.ndarray): The computed 2D PSF.
                - x_edges (be.ndarray): Bin edges for the x-axis.
                - y_edges (be.ndarray): Bin edges for the y-axis.
        """
        if self.ray_intersections_x is None or self.ray_intersections_y is None:
            raise RuntimeError("Ray intersection data has not been generated.")

        # Determine bounds for the histogram dynamically
        min_x = be.min(self.ray_intersections_x)
        max_x = be.max(self.ray_intersections_x)
        min_y = be.min(self.ray_intersections_y)
        max_y = be.max(self.ray_intersections_y)

        # Handle cases where all rays land at the same point (e.g. perfect lens on axis)
        if min_x == max_x:
            # Add a small delta if min and max are the same
            # Use a small fixed physical size, e.g. 0.001 mm
            delta_x = 0.001
            min_x -= delta_x / 2
            max_x += delta_x / 2
        if min_y == max_y:
            delta_y = 0.001
            min_y -= delta_y / 2
            max_y += delta_y / 2

        # Create the 2D histogram
        psf_image, x_edges, y_edges = be.histogram2d(
            self.ray_intersections_x,
            self.ray_intersections_y,
            bins=self.grid_size,
            range=[[min_x, max_x], [min_y, max_y]],
            weights=self.ray_intensities,
        )

        # Normalize PSF
        intensity_sum = be.sum(self.ray_intensities)
        if intensity_sum > 0:
            psf_image = (psf_image / intensity_sum) * 100
        else:
            # Handle case with no rays or all rays outside bounds (empty psf_image)
            psf_image = be.zeros_like(psf_image)

        return psf_image, x_edges, y_edges

    def _get_psf_units(self, image):
        """Calculates the physical extent (units) of the PSF image for plotting.

        This method is called by `BasePSF.view()` to determine axis labels.
        The units are determined by the units of the ray intersection data,
        which are typically in millimeters (mm). The plot will be in micrometers (µm).

        Args:
            image (be.ndarray): The PSF image data (often a
                zoomed/cropped version from `BasePSF.view`). Its shape is used
                to determine the total extent for labeling.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the physical
            total width and total height of the PSF image area, in micrometers.
        """
        width_mm = be.to_numpy(self.x_edges[-1] - self.x_edges[0])
        height_mm = be.to_numpy(self.y_edges[-1] - self.y_edges[0])

        # Convert mm to µm
        width_um = width_mm * 1000
        height_um = height_mm * 1000

        return width_um, height_um
