"""Fast Fourier Transform Modulation Transfer Function (FFTMTF) Module.

This module provides the FFTMTF class for computing the MTF
of an optical system using FFT techniques.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.psf.fft import FFTPSF, calculate_grid_size

from .base import BaseMTF


class FFTMTF(BaseMTF):
    """Fast Fourier Transform Modulation Transfer Function (FFTMTF) class.

    This class calculates and visualizes the Modulation Transfer Function (MTF)
    of an optic using the Fast Fourier Transform (FFT) method.

    Args:
        optic (Optic): The optic for which to calculate the MTF.
        fields (str or list, optional): The field coordinates for which to
            calculate the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength of light to use
            for the MTF calculation. Defaults to 'primary'.
        num_rays (int, optional): The number of rays to use for the PSF
            calculation, which is then used for MTF. Defaults to 128.
        grid_size (int or None, optional): The size of the grid used for the PSF/MTF
            calculation. If `None`, the grid size will be calculated from `num_rays`
            as documented in `optiland.psf.fft.FFTPSF`. Defaults to `None`.
        max_freq (str or float, optional): The maximum frequency for the MTF
            calculation. Defaults to 'cutoff'.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray", "centroid_sphere", and "best_fit_sphere".
            Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the strategy.

    Attributes:
        num_rays (int): The number of rays used for the MTF calculation.
        grid_size (int): The size of the grid used for the MTF calculation.
        max_freq (float): The maximum frequency for the MTF calculation.
        FNO (float): The F-number of the optic.
        psf (list): List of PSF data for each field.
        mtf (list): List of MTF data ([tangential, sagittal]) for each field.
        freq (be.ndarray): Array of frequency points for the MTF curve.
    """

    def __init__(
        self,
        optic,
        fields: str | list = "all",
        wavelength: str | float = "primary",
        num_rays=128,
        grid_size=None,
        max_freq="cutoff",
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        if grid_size is None:
            self.num_rays, self.grid_size = calculate_grid_size(num_rays)
        else:
            self.num_rays = num_rays
            self.grid_size = grid_size

        super().__init__(optic, fields, wavelength, strategy, remove_tilt, **kwargs)

        self.FNO = self._get_fno()

        if max_freq == "cutoff":
            self.max_freq = 1 / (self.resolved_wavelength * 1e-3 * self.FNO)
        else:
            self.max_freq = max_freq

        self.freq = be.arange(self.grid_size // 2) * self._get_mtf_units()

    def _calculate_psf(self):
        """Calculates and stores the Point Spread Function (PSF)

        This method uses the resolved field points and wavelength from BaseMTF.
        """
        self.psf = [
            FFTPSF(
                self.optic,
                field,
                self.resolved_wavelength,
                self.num_rays,
                self.grid_size,
                self.strategy,
                self.remove_tilt,
                **self.strategy_kwargs,
            ).psf
            for field in self.resolved_fields
        ]

    def _plot_field_mtf(self, ax, field_index, mtf_field_data, color):
        """Plots the MTF data for a single field for FFTMTF.

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
            tangential = data[self.grid_size // 2 :, self.grid_size // 2]
            sagittal = data[self.grid_size // 2, self.grid_size // 2 :]
            mtf.append([tangential / be.max(tangential), sagittal / be.max(sagittal)])
        return mtf

    def _get_mtf_units(self):
        """Calculate the MTF units.

        Returns:
            float: The MTF units calculated based on the grid size, number
                of rays, wavelength (from BaseMTF), and FNO.
        """
        dx = 1 / ((self.num_rays - 1) * self.resolved_wavelength * 1e-3 * self.FNO)

        return dx
