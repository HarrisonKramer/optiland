"""Geometric Modulation Transfer Function (MTF) Module

This module provides the GeometricMTF class for calculating the MTF based
on ray trace data (spot diagrams).

Kramer Harrison, 2025 (Assumed author based on other files)
"""

import optiland.backend as be
from optiland.analysis import SpotDiagram
from optiland.mtf.base import BaseMTF


class GeometricMTF(BaseMTF):
    """Geometric Modulation Transfer Function (MTF) based on spot diagrams.

    This class calculates the MTF of an optical system from the geometric
    distribution of ray intersection points on the image plane. The calculation
    is based on the method described in Smith, Modern Optical Engineering,
    3rd edition, Section 11.9.

    It inherits common setup and plotting functionalities from `BaseMTF`.

    Args:
        optic (Optic): The optical system.
        fields (str or list): Field points for MTF calculation.
        wavelength (str or float): Wavelength for MTF calculation.
        num_rays (int, optional): Number of rays for spot diagram. Defaults to 1000.
                               Note: BaseMTF default is 128, but geometric
                               benefits from more rays.
        distribution (str, optional): Ray distribution for spot diagram.
                                  Defaults to 'uniform'.
        num_points (int, optional): Number of points for MTF curve.
                                Defaults to BaseMTF's default (256).
        max_freq (str or float, optional): Maximum frequency.
                                       Defaults to BaseMTF's default ('cutoff').
        scale (bool, optional): Whether to scale the geometric MTF by the
                              diffraction-limited MTF. Defaults to True.

    Attributes:
        scale (bool): If True, scales the computed MTF by the diffraction limit.
        distribution (str): The ray distribution used for SpotDiagram.
        spot_diagram_data (list): Raw data from `SpotDiagram` used for calculations.
                                 Stored after running SpotDiagram.
    """

    def __init__(
        self,
        optic,
        fields,
        wavelength,
        num_rays=1000,  # Geometric benefits from more rays than default BaseMTF
        distribution="uniform",
        num_points=256,  # Consistent with BaseMTF default
        max_freq="cutoff",
        scale=True,
    ):
        super().__init__(
            optic=optic,
            fields=fields,
            wavelength=wavelength,
            max_freq=max_freq,
            num_points=num_points,
            num_rays=num_rays,  # Pass num_rays to BaseMTF, it stores it
        )
        # Note: self.num_rays from BaseMTF will hold the value passed here.
        # self.grid_size is also inherited but not directly used by GeometricMTF.

        self.scale = scale
        self.distribution = distribution  # Store for _generate_mtf_data

        # Generate and store spot diagram data needed for MTF calculation
        # BaseMTF sets self.wavelength (single float) and self.fields (list of tuples)
        # SpotDiagram expects wavelengths as a list.
        spot_diagram_calculator = SpotDiagram(
            self.optic,
            fields=self.fields,  # Use parsed fields from BaseMTF
            wavelengths=[self.wavelength],  # Use parsed wavelength from BaseMTF
            num_rays=self.num_rays,  # Use num_rays stored by BaseMTF
            distribution=self.distribution,
        )
        self.spot_diagram_data = spot_diagram_calculator.data
        # self.data from SpotDiagram is a list of lists (fields, then wavelengths).
        # For GeometricMTF, we have one wavelength.
        # So, self.spot_diagram_data[field_idx][0] will give the SpotDataItem.

        # Compute MTF data upon initialization by calling the main method
        self.mtf_data = self._generate_mtf_data()

    def _compute_single_mtf_profile(self, ray_coords, scale_factor):
        """Computes one MTF profile (e.g., tangential or sagittal) from ray coordinates.

        Args:
            ray_coords (be.ndarray): Array of ray intersection coordinates
                (e.g., x or y).
                                     Coordinates are typically in mm.
            scale_factor (be.ndarray or float): Factor to scale the MTF by. This is
                                              either the diffraction-limited MTF
                                              (if self.scale is True) or 1.0.

        Returns:
            be.ndarray: The computed MTF profile for the given ray coordinates.
        """
        # MTF is computed over self.freq, which is from BaseMTF (cycles/mm)
        # self.num_points is the number of points for the MTF curve (length of
        # self.freq)

        if ray_coords is None or len(ray_coords) == 0:  # Handle cases with no rays
            return be.zeros_like(self.freq)

        # Filter out NaN or inf values from ray_coords, as they can break histogram
        ray_coords_clean = ray_coords[be.isfinite(ray_coords)]

        if (
            len(ray_coords_clean) == 0
        ):  # Handle cases where all rays were NaN/inf or filtered out
            return be.zeros_like(self.freq)

        # Determine histogram range dynamically
        min_coord = be.min(ray_coords_clean)
        max_coord = be.max(ray_coords_clean)

        if min_coord == max_coord:  # All rays at the same point
            # Create a small range for histogram to avoid errors
            # Use a small fixed physical size, e.g. 1 micron (0.001 mm)
            delta = 0.001
            min_coord -= delta / 2
            max_coord += delta / 2

        # Histogram of ray coordinates
        # Using self.num_points as number of bins for the histogram.
        # Smith's book implies the histogram resolution can affect results.
        # More bins might offer better precision for the histogram's shape.
        # For now, using self.num_points to match original behavior.
        num_hist_bins = self.num_points

        A, edges = be.histogram(
            ray_coords_clean, bins=num_hist_bins, range=[min_coord, max_coord]
        )
        x_hist = (edges[1:] + edges[:-1]) / 2.0  # Bin centers

        # Ensure dx is scalar, not array, if x_hist has only one element
        if len(x_hist) > 1:
            dx = x_hist[1] - x_hist[0]  # Bin width
        elif len(x_hist) == 1:  # Single bin from histogram
            dx = edges[1] - edges[0]  # Width of that single bin
        else:  # No bins, e.g. if ray_coords_clean was empty or range was zero somehow
            return be.zeros_like(self.freq)

        mtf_profile = be.zeros_like(
            self.freq
        )  # Using be.zeros_like for backend compatibility

        # Sum of amplitudes (total number of rays in histogram) times bin width
        # This is sum(A_i * dx_i), effectively integrating the histogram.
        # Represents the "volume" under the histogram.
        sum_A_dx = be.sum(A * dx)
        if (
            sum_A_dx == 0
        ):  # Avoid division by zero if no rays in histogram or dx is zero
            return mtf_profile  # Returns all zeros

        # Spatial frequencies (v_freq) are in self.freq (cycles/mm)
        # Ray coordinates (x_hist) are in mm
        # The argument of cos/sin is 2 * pi * v * x, which is unitless.
        for k_freq, v_freq in enumerate(self.freq):
            # Cosine and Sine transforms
            cos_terms = be.cos(2 * be.pi * v_freq * x_hist)
            sin_terms = be.sin(2 * be.pi * v_freq * x_hist)

            # Numerators of the transform integrals
            # Sum ( A_i * cos(2*pi*v*x_i) * dx_i )
            Ac_numerator = be.sum(A * cos_terms * dx)
            As_numerator = be.sum(A * sin_terms * dx)

            # Normalized transforms
            Ac = Ac_numerator / sum_A_dx
            As = As_numerator / sum_A_dx

            mtf_profile[k_freq] = be.sqrt(Ac**2 + As**2)

        return mtf_profile * scale_factor

    def _generate_mtf_data(self):
        """Generates the MTF data for each field point.

        This implements the abstract method from BaseMTF.
        It calculates tangential and sagittal MTF profiles using ray spot data.
        The convention used is:
        - Tangential MTF: Computed from the y-coordinates of the spot diagram.
        - Sagittal MTF: Computed from the x-coordinates of the spot diagram.
        This is a common convention, particularly for fields along the x or y axes.
        For off-axis fields, these correspond to projections of the spot distribution.

        Returns:
            list: A list of [tangential_mtf, sagittal_mtf] pairs, one for each
                  field defined in self.fields.
        """
        if self.spot_diagram_data is None:
            # This should not happen if __init__ completed successfully.
            raise RuntimeError(
                "Spot diagram data has not been generated. This indicates an issue in "
                "initialization."
            )

        # Determine scale factor: diffraction limit or 1.0
        if self.scale:  # noqa: SIM108
            # self.diff_limited_mtf is already calculated by BaseMTF
            scale_factor = self.diff_limited_mtf
        else:
            scale_factor = 1.0  # No scaling

        calculated_mtfs = []
        # self.spot_diagram_data is list (per field) of list (per wavelength)
        # We have one wavelength, so data[field_idx][0]
        for field_idx in range(len(self.fields)):
            # SpotDataItem for current field, first (and only) wavelength
            spot_data_item = self.spot_diagram_data[field_idx][0]

            # Ray coordinates from SpotDiagram (typically in mm).
            # SpotDiagram.x and .y are usually relative to the chief ray for that field.

            # Convention:
            # Tangential MTF: computed from y-coordinates of spots.
            # Sagittal MTF: computed from x-coordinates of spots.
            # TODO: Implement more sophisticated Tangential/Sagittal orientation based
            # on field angle
            # for more accurate T/S definitions for arbitrary field points.
            # For now, this matches common practice and previous behavior.

            coords_for_tangential = spot_data_item.y
            coords_for_sagittal = spot_data_item.x

            mtf_t = self._compute_single_mtf_profile(
                coords_for_tangential, scale_factor
            )
            mtf_s = self._compute_single_mtf_profile(coords_for_sagittal, scale_factor)

            calculated_mtfs.append([mtf_t, mtf_s])

        return calculated_mtfs
