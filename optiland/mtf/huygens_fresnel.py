"""Huygens Modulation Transfer Function (HuygensMTF) Module.

This module provides the HuygensMTF class for computing the MTF
of an optical system using Huygens-Fresnel PSF.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.psf.huygens_fresnel import HuygensPSF

from .base import BaseMTF


class HuygensMTF(BaseMTF):
    """Huygens Modulation Transfer Function (HuygensMTF) class.

    This class calculates and visualizes the Modulation Transfer Function (MTF)
    of an optic using a Point Spread Function (PSF) derived from the
    Huygens-Fresnel principle.

    Note: This class currently only supports the 'numpy' backend due to the
    underlying HuygensPSF implementation.

    Args:
        optic (Optic): The optic for which to calculate the MTF.
        fields (str or list, optional): The field coordinates for which to
            calculate the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength of light to use
            for the MTF calculation. Defaults to 'primary'.
        num_rays (int, optional): The number of rays to use for the HuygensPSF
            calculation along one dimension of the pupil grid. Defaults to 128.
        image_size (int, optional): The size of the image grid for PSF
            calculation and subsequent MTF. Defaults to 128.
        max_freq (str or float, optional): The maximum frequency for the MTF
            calculation in cycles/mm. If 'cutoff', it's determined by the
            diffraction limit (1 / (lambda * FNO)). Defaults to 'cutoff'.

    Attributes:
        num_rays (int): The number of rays used for the PSF calculation.
        image_size (int): The size of the grid used for the PSF/MTF calculation.
        max_freq (float): The maximum frequency for the MTF calculation (cycles/mm).
        FNO (float): The F-number of the optic.
        psf_data (list): List of 2D PSF data arrays for each field.
        psf_instances (list): List of HuygensPSF instances for each field.
        mtf (list): List of MTF data ([tangential, sagittal]) for each field.
        freq (be.ndarray): Array of frequency points for the MTF curve (cycles/mm).
    """

    def __init__(
        self,
        optic,
        fields: str | list = "all",
        wavelength: str | float = "primary",
        num_rays=128,
        image_size=128,
        max_freq="cutoff",
    ):
        if be.get_backend() != "numpy":
            raise ValueError(
                "HuygensMTF only supports the 'numpy' backend due to "
                "the underlying HuygensPSF implementation."
            )

        self.num_rays = num_rays
        self.image_size = image_size
        self.psf_instances = []

        super().__init__(optic, fields, wavelength)

        self.FNO = self._get_fno()

        if max_freq == "cutoff":
            # wavelength in um, FNO is unitless. max_freq in cycles/mm
            self.max_freq = 1 / (self.resolved_wavelength * 1e-3 * self.FNO)
        else:
            self.max_freq = max_freq

        self.freq = be.arange(self.image_size // 2) * self._get_mtf_units()

    def _calculate_psf(self):
        """Calculates and stores the Point Spread Functions (PSFs).

        This method uses the resolved field points and wavelength from BaseMTF.
        It populates `self.psf_data` with 2D PSF arrays and `self.psf_instances`
        with the `HuygensPSF` objects.
        """
        self.psf_data = []
        self.psf_instances = []  # Reset or initialize
        for field_coord in self.resolved_fields:
            psf_calculator = HuygensPSF(
                optic=self.optic,
                field=field_coord,  # Single field tuple
                wavelength=self.resolved_wavelength,  # Single wavelength value
                num_rays=self.num_rays,
                image_size=self.image_size,
                oversample=2.0,  # oversampling ratio with respect to the optical cutoff
            )
            self.psf_data.append(psf_calculator.psf)
            self.psf_instances.append(psf_calculator)

    def _generate_mtf_data(self):
        """Generates the MTF data for each field from the calculated PSFs.

        The calculation involves:
        1. Taking the 2D FFT of each PSF.
        2. Shifting the zero-frequency component to the center.
        3. Taking the absolute magnitude.
        4. Extracting tangential and sagittal slices.
        5. Normalizing these slices.

        Returns:
            list: A list of MTF data for each field. Each item is a list
                  containing the normalized tangential and sagittal MTF arrays.
        """
        mtf_results = []
        for psf_array in self.psf_data:
            otf = be.fft.fftshift(be.fft.fft2(psf_array))
            mtf_abs = be.abs(otf)

            # Determine center for slicing (HuygensPSF output is image_size x
            # image_size).
            center_idx = self.image_size // 2

            # Extract tangential and sagittal MTF
            # Tangential: Column at center_idx, from center_idx downwards
            tangential_mtf = mtf_abs[center_idx:, center_idx]
            # Sagittal: Row at center_idx, from center_idx rightwards
            sagittal_mtf = mtf_abs[center_idx, center_idx:]

            # Ensure they are 1D arrays of expected length (image_size // 2)
            tangential_mtf = tangential_mtf[: self.image_size // 2]
            sagittal_mtf = sagittal_mtf[: self.image_size // 2]

            # Normalize MTF by its DC component (value at zero frequency)
            # DC component of MTF (OTF magnitude) is mtf_abs[center_idx, center_idx].
            dc_value = mtf_abs[center_idx, center_idx]
            if dc_value == 0:  # Avoid division by zero
                # This would mean the PSF sum is zero, highly unlikely for valid inputs.
                norm_tangential = be.zeros_like(tangential_mtf)
                norm_sagittal = be.zeros_like(sagittal_mtf)
            else:
                norm_tangential = tangential_mtf / dc_value
                norm_sagittal = sagittal_mtf / dc_value

            # Ensure MTF <= 1 after normalization (can exceed due to numerical
            # precision).
            norm_tangential = be.clip(norm_tangential, 0.0, 1.0)
            norm_sagittal = be.clip(norm_sagittal, 0.0, 1.0)

            mtf_results.append([norm_tangential, norm_sagittal])

        return mtf_results

    def _plot_field_mtf(self, ax, field_index, mtf_field_data, color):
        """Plots the MTF data (tangential and sagittal) for a single field.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            field_index (int): The index of the current field in self.resolved_fields.
            mtf_field_data (list): A list containing normalized tangential and sagittal
                                   MTF data (be.ndarray) for the field.
            color (str): The color to use for plotting this field.
        """
        current_field_label_info = self.resolved_fields[field_index]

        # Ensure freq array matches length of MTF data after potential truncation
        # in __init__ and slicing in _generate_mtf_data
        num_mtf_points = mtf_field_data[0].shape[0]
        freq_for_plot = self.freq[:num_mtf_points]

        # Plot tangential MTF
        ax.plot(
            be.to_numpy(freq_for_plot),
            be.to_numpy(mtf_field_data[0]),  # Tangential data
            label=(
                f"Hx: {current_field_label_info[0]:.1f}, "
                f"Hy: {current_field_label_info[1]:.1f}, Tangential"
            ),
            color=color,
            linestyle="-",
        )
        # Plot sagittal MTF
        ax.plot(
            be.to_numpy(freq_for_plot),
            be.to_numpy(mtf_field_data[1]),  # Sagittal data
            label=(
                f"Hx: {current_field_label_info[0]:.1f}, "
                f"Hy: {current_field_label_info[1]:.1f}, Sagittal"
            ),
            color=color,
            linestyle="--",
        )

    def _get_mtf_units(self):
        """Calculate the MTF frequency step (spatial frequency units).

        The frequency unit is cycles per mm.
        It's determined by the pixel pitch of the PSF image and the total
        number of pixels (image_size). The maximum frequency that can be
        represented is related to Nyquist, 1 / (2 * pixel_pitch).
        The frequency step df = 1 / (image_size * pixel_pitch).

        Returns:
            float: The frequency step for MTF calculation (cycles/mm).
        """
        # Use pixel_pitch from the first PSF instance (should be consistent)
        # HuygensPSF stores pixel_pitch in mm.
        pixel_pitch_mm = self.psf_instances[0].pixel_pitch
        if pixel_pitch_mm is None or pixel_pitch_mm == 0:
            raise ValueError("Pixel pitch from HuygensPSF is invalid.")

        # Total extent of the PSF image in mm is image_size * pixel_pitch_mm
        # The fundamental frequency (step) in the DFT is 1 / total_extent
        df = 1.0 / (self.image_size * pixel_pitch_mm)
        return df
