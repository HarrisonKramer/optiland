"""Huygens Modulation Transfer Function (MTF) Module.

This module provides the HuygensMTF class for computing the MTF
of an optical system using the Huygens-Fresnel PSF.

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.psf.huygens_fresnel import HuygensPSF

from .base import BaseMTF


class HuygensMTF(BaseMTF):
    """Huygens-Fresnel based Modulation Transfer Function (MTF) class.

    This class calculates and visualizes the Modulation Transfer Function (MTF)
    of an optic using a PSF calculated via the Huygens-Fresnel principle.

    Args:
        optic (Optic): The optic for which to calculate the MTF.
        fields (str or list, optional): The field coordinates for which to
            calculate the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength of light to use
            for the MTF calculation. Defaults to 'primary'.
        num_rays (int, optional): The number of rays to use for the PSF
            calculation, which is then used for MTF. Defaults to 32.
        image_size (int, optional): The size of the grid used for the PSF/MTF
            calculation. Defaults to 32.
        max_freq (str or float, optional): The maximum frequency for the MTF
            calculation. Defaults to 'cutoff'.

    Attributes:
        num_rays (int): The number of rays used for the MTF calculation.
        image_size (int): The size of the grid used for the MTF calculation.
        max_freq (float): The maximum frequency for the MTF calculation.
        FNO (float): The F-number of the optic.
        psf (list): List of PSF data for each field.
        mtf (list): List of MTF data ([tangential, sagittal]) for each field.
        freq (be.ndarray): Array of frequency points for the MTF curve.
    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelength="primary",
        num_rays=32,
        image_size=32,
        max_freq="cutoff",
    ):
        if be.get_backend() != "numpy":
            raise ValueError("HuygensMTF only supports the numpy backend.")

        self.num_rays = num_rays
        self.image_size = image_size
        self.max_freq_arg = max_freq

        super().__init__(optic, fields, wavelength)

        self.FNO = self._get_fno()

        if self.max_freq_arg == "cutoff":
            # The actual max_freq depends on pixel pitch from HuygensPSF,
            # which is calculated during _calculate_psf.
            # We calculate it after the PSF is computed.
            pass
        else:
            self.max_freq = self.max_freq_arg

        self.freq = be.arange(self.image_size // 2) * self._get_mtf_units()

    def _calculate_psf(self):
        """Calculates and stores the Point Spread Function (PSF)

        This method uses the resolved field points and wavelength from BaseMTF.
        """
        self.psf_calculators = [
            HuygensPSF(
                self.optic,
                field,
                self.resolved_wavelength,
                self.num_rays,
                self.image_size,
            )
            for field in self.resolved_fields
        ]
        self.psf = [psf_calc.psf for psf_calc in self.psf_calculators]

        # Now that PSF is calculated, we can determine the frequency scale
        if self.max_freq_arg == "cutoff":
            pixel_pitch = self.psf_calculators[0].pixel_pitch  # in mm
            self.max_freq = 1 / (2 * pixel_pitch)

        self.freq = be.arange(self.image_size // 2) * self._get_mtf_units()

    def _generate_mtf_data(self):
        """Generates the MTF data for each field.

        The calculation is based on the PSF, which is calculated during
        construction of the class.

        Returns:
            list: A list of MTF data for each field. Each MTF data is a list
                containing the tangential and sagittal MTF values.
        """
        mtf_data = [be.abs(be.fft.fftshift(be.fft.fft2(psf))) for psf in self.psf]
        mtf = []
        for data in mtf_data:
            center = self.image_size // 2
            tangential = data[center:, center]
            sagittal = data[center, center:]

            # Normalize to have DC=1
            tangential = tangential / tangential[0]
            sagittal = sagittal / sagittal[0]

            mtf.append([tangential, sagittal])
        return mtf

    def _plot_field_mtf(self, ax, field_index, mtf_field_data, color):
        """Plots the MTF data for a single field for HuygensMTF.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            field_index (int): The index of the current field in self.resolved_fields.
            mtf_field_data (list): A list containing tangential and sagittal
                                   MTF data (be.ndarray) for the field.
            color (str): The color to use for plotting this field.
        """
        current_field_label_info = self.resolved_fields[field_index]

        # Plot tangential MTF
        ax.plot(
            be.to_numpy(self.freq),
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
            be.to_numpy(self.freq),
            be.to_numpy(mtf_field_data[1]),  # Sagittal data
            label=(
                f"Hx: {current_field_label_info[0]:.1f}, "
                f"Hy: {current_field_label_info[1]:.1f}, Sagittal"
            ),
            color=color,
            linestyle="--",
        )

    def _get_mtf_units(self):
        """Calculate the MTF units (frequency step).

        Returns:
            float: The MTF frequency step in cycles/mm.
        """
        if not hasattr(self, "psf_calculators"):
            # This can happen if called from __init__ before _calculate_psf
            return 0

        pixel_pitch = self.psf_calculators[0].pixel_pitch  # in mm
        return 1 / (self.image_size * pixel_pitch)

    def _get_fno(self):
        """Calculate the effective F-number (FNO) of the optical system.

        Applies a correction if the object is finite.
        Uses self.optic from BaseMTF.

        Returns:
            float: The effective F-number of the optical system.
        """
        FNO = self.optic.paraxial.FNO()

        if not self.optic.object_surface.is_infinite:
            D = self.optic.paraxial.XPD()
            p = D / self.optic.paraxial.EPD()
            m = self.optic.paraxial.magnification()
            FNO = FNO * (1 + be.abs(m) / p)

        return FNO
