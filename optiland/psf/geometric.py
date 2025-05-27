"""Geometric Point Spread Function (PSF) Module

This module provides functionality for simulating the Point Spread Function (PSF)
of optical systems based on geometric ray tracing (spot diagrams).

Kramer Harrison, 2025 (Assumed author based on other files)
"""

import optiland.backend as be
from optiland.psf.base import BasePSF
from optiland.analysis import SpotDiagram # SpotDiagram is used to get ray data


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

    def __init__(self, optic, field, wavelength, num_rays=10000, grid_size=256, distribution="uniform"):
        super().__init__(
            optic=optic, field=field, wavelength=wavelength, num_rays=num_rays
        )
        self.grid_size = grid_size

        # We need to get the ray data first using SpotDiagram
        # SpotDiagram takes a list of wavelengths.
        self.spot_data = SpotDiagram(
            optic=optic,
            fields=[field], # SpotDiagram expects a list of fields
            wavelengths=[wavelength],
            num_rays=num_rays,
            distribution=distribution
        )
        # SpotDiagram stores data in a list of lists (fields, then wavelengths)
        # For GeometricPSF, we have one field and one wavelength.
        self.ray_intersections_x = self.spot_data.data[0][0].x
        self.ray_intersections_y = self.spot_data.data[0][0].y

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

        # Determine bounds for the histogram dynamically if not specified
        # This ensures the main concentration of spots is captured.
        min_x = be.min(self.ray_intersections_x)
        max_x = be.max(self.ray_intersections_x)
        min_y = be.min(self.ray_intersections_y)
        max_y = be.max(self.ray_intersections_y)

        # Handle cases where all rays land at the same point (e.g. perfect lens on axis)
        if min_x == max_x:
            # Add a small delta if min and max are the same
            # Use a small fixed physical size, e.g. 1 micron, if units are mm
            delta_x = 0.001 # 1 micrometer in mm
            min_x -= delta_x / 2
            max_x += delta_x / 2
        if min_y == max_y:
            delta_y = 0.001 # 1 micrometer in mm
            min_y -= delta_y / 2
            max_y += delta_y / 2
            
        # Create the 2D histogram
        try:
            psf_image, x_edges, y_edges = be.histogram2d(
                self.ray_intersections_x,
                self.ray_intersections_y,
                bins=self.grid_size,
                range=[[min_x, max_x], [min_y, max_y]]
            )
        except AttributeError:
            # Fallback to numpy if backend doesn't support histogram2d
            np_psf_image, np_x_edges, np_y_edges = be.np.histogram2d(
                be.to_numpy(self.ray_intersections_x),
                be.to_numpy(self.ray_intersections_y),
                bins=self.grid_size,
                range=[[be.to_numpy(min_x), be.to_numpy(max_x)], [be.to_numpy(min_y), be.to_numpy(max_y)]]
            )
            psf_image = be.array(np_psf_image)
            x_edges = be.array(np_x_edges)
            y_edges = be.array(np_y_edges)

        # Normalize PSF so that peak is 100
        if be.max(psf_image) > 0:
            psf_image = (psf_image / be.max(psf_image)) * 100
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
        # x_edges and y_edges are the bin edges from the histogram.
        # The total extent is the difference between the last and first edge.
        # SpotDiagram data is in mm, convert to µm for display.
        
        if self.x_edges is None or self.y_edges is None:
            # This case should ideally not be reached if __init__ completes successfully.
            # Defaulting to pixel size if edges are not set, though this is not ideal.
            return be.to_numpy(image.shape[1]), be.to_numpy(image.shape[0])

        physical_width_mm = be.to_numpy(self.x_edges[-1] - self.x_edges[0])
        physical_height_mm = be.to_numpy(self.y_edges[-1] - self.y_edges[0])

        # Convert mm to µm
        physical_width_um = physical_width_mm * 1000
        physical_height_um = physical_height_mm * 1000
        
        return physical_width_um, physical_height_um

    def strehl_ratio(self):
        """Computes the Strehl ratio of the PSF.

        For a geometric PSF, the Strehl ratio is not typically defined in the
        same way as for diffraction-based PSFs. This method returns the peak
        value of the normalized geometric PSF (which is 1.0 by construction, 
        as it's normalized to 100 and then divided by 100 by BasePSF's strehl_ratio).

        Returns:
            float: The Strehl ratio.
        """
        return super().strehl_ratio()

```
