"""Modulation Transfer Function (MTF) Module

This module provides various classes for the computation of the modulation
transfer function (MTF) of an optical system.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.analysis import SpotDiagram
from optiland.psf import FFTPSF


class GeometricMTF(SpotDiagram):
    """Smith, Modern Optical Engineering 3rd edition, Section 11.9

    This class represents the Geometric MTF (Modulation Transfer Function) of
    an optical system. It inherits from the SpotDiagram class.

    Args:
        optic (Optic): The optical system for which to calculate the MTF.
        fields (str or list, optional): The field points at which to calculate
            the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength at which to
            calculate the MTF. Defaults to 'primary'.
        num_rays (int, optional): The number of rays to trace for each field
            point. Defaults to 100.
        distribution (str, optional): The distribution of rays within each
            field point. Defaults to 'uniform'.
        num_points (int, optional): The number of points to sample in the MTF
            curve. Defaults to 256.
        max_freq (str or float, optional): The maximum frequency to consider
            in the MTF curve. Defaults to 'cutoff'.
        scale (bool, optional): Whether to scale the MTF curve using the
            diffraction-limited curve. Defaults to True.

    Attributes:
        num_points (int): The number of points to sample in the MTF curve.
        scale (bool): Whether to scale the MTF curve.
        max_freq (float): The maximum frequency to consider in the MTF curve.
        freq (ndarray): The frequency values for the MTF curve.
        mtf (list): The MTF data for each field point.
        diff_limited_mtf (ndarray): The diffraction-limited MTF curve.

    Methods:
        view(figsize=(12, 4), add_reference=False): Plots the MTF curve.
        _generate_mtf_data(): Generates the MTF data for each field point.
        _compute_field_data(xi, v, scale_factor): Computes the MTF data for a
            given field point.
        _plot_field(ax, mtf_data, field, color): Plots the MTF data for a
            given field point.

    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelength="primary",
        num_rays=100,
        distribution="uniform",
        num_points=256,
        max_freq="cutoff",
        scale=True,
    ):
        self.num_points = num_points
        self.scale = scale

        if wavelength == "primary":
            wavelength = optic.primary_wavelength
        if max_freq == "cutoff":
            # wavelength must be converted to mm for frequency units cycles/mm
            self.max_freq = 1 / (wavelength * 1e-3 * optic.paraxial.FNO())

        super().__init__(optic, fields, [wavelength], num_rays, distribution)

        self.freq = be.linspace(0, self.max_freq, num_points)
        self.mtf, self.diff_limited_mtf = self._generate_mtf_data()

    def view(self, figsize=(12, 4), add_reference=False):
        """Plots the MTF curve.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (12, 4).
            add_reference (bool, optional): Whether to add the diffraction
                limit reference curve. Defaults to False.

        """
        _, ax = plt.subplots(figsize=figsize)

        for k, data in enumerate(self.mtf):
            self._plot_field(ax, data, self.fields[k], color=f"C{k}")

        if add_reference:
            ax.plot(self.freq, self.diff_limited_mtf, "k--", label="Diffraction Limit")

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim([0, self.max_freq])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Frequency (cycles/mm)", labelpad=10)
        ax.set_ylabel("Modulation", labelpad=10)
        plt.tight_layout()
        plt.show()

    def _generate_mtf_data(self):
        """Generates the MTF data for each field point.

        Returns:
            tuple: A tuple containing the MTF data for each field point and
                the scale factor.

        """
        if self.scale:
            phi = be.arccos(self.freq / self.max_freq)
            scale_factor = 2 / be.pi * (phi - be.cos(phi) * be.sin(phi))
        else:
            scale_factor = 1

        mtf = []  # TODO: add option for polychromatic MTF
        for field_data in self.data:
            xi, yi = field_data[0][0], field_data[0][1]
            mtf.append(
                [
                    self._compute_field_data(yi, self.freq, scale_factor),
                    self._compute_field_data(xi, self.freq, scale_factor),
                ],
            )
        return mtf, scale_factor

    def _compute_field_data(self, xi, v, scale_factor):
        """Computes the MTF data for a given field point.

        Args:
            xi (ndarray): The coordinate values (x or y) of the field point.
            v (ndarray): The frequency values for the MTF curve.
            scale_factor (float): The scale factor for the MTF curve.

        Returns:
            ndarray: The MTF data for the field point.

        """
        A, edges = be.histogram(xi, bins=self.num_points + 1)
        x = (edges[1:] + edges[:-1]) / 2
        dx = x[1] - x[0]

        mtf = be.zeros_like(v)
        for k in range(len(v)):
            Ac = be.sum(A * be.cos(2 * be.pi * v[k] * x) * dx) / be.sum(A * dx)
            As = be.sum(A * be.sin(2 * be.pi * v[k] * x) * dx) / be.sum(A * dx)

            mtf[k] = be.sqrt(Ac**2 + As**2)

        return mtf * scale_factor

    def _plot_field(self, ax, mtf_data, field, color):
        """Plots the MTF data for a given field point.

        Args:
            ax (Axes): The matplotlib axes object.
            mtf_data (ndarray): The MTF data for the field point.
            field (tuple): The field point coordinates.
            color (str): The color of the plotted lines.

        """
        ax.plot(
            self.freq,
            mtf_data[0],
            label=f"Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Tangential",
            color=color,
            linestyle="-",
        )
        ax.plot(
            self.freq,
            mtf_data[1],
            label=f"Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Sagittal",
            color=color,
            linestyle="--",
        )


class FFTMTF:
    """Fast Fourier Transform Modulation Transfer Function (FFTMTF) class.

    This class calculates and visualizes the Modulation Transfer Function (MTF)
    of an optic using the Fast Fourier Transform (FFT) method.

    Args:
        optic (Optic): The optic for which to calculate the MTF.
        fields (str or list, optional): The field coordinates for which to
            calculate the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength of light to use
            for the MTF calculation. Defaults to 'primary'.
        num_rays (int, optional): The number of rays to use for the MTF
            calculation. Defaults to 128.
        grid_size (int, optional): The size of the grid used for the MTF
            calculation. Defaults to 1024.
        max_freq (str or float, optional): The maximum frequency for the MTF
            calculation. Defaults to 'cutoff'.

    Attributes:
        optic (Optic): The optic for which the MTF is calculated.
        fields (list): The field coordinates for which the MTF is calculated.
        wavelength (float): The wavelength of light used for the MTF
            calculation.
        num_rays (int): The number of rays used for the MTF calculation.
        grid_size (int): The size of the grid used for the MTF calculation.
        max_freq (float): The maximum frequency for the MTF calculation.
        FNO (float): The F-number of the optic.

    Methods:
        view(figsize=(12, 4), add_reference=False): Visualizes the MTF.

    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelength="primary",
        num_rays=128,
        grid_size=1024,
        max_freq="cutoff",
    ):
        self.optic = optic
        self.max_freq = max_freq
        self.fields = fields
        self.wavelength = wavelength
        self.num_rays = num_rays
        self.grid_size = grid_size

        self.FNO = self._get_fno()

        if self.fields == "all":
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelength == "primary":
            self.wavelength = optic.primary_wavelength

        if max_freq == "cutoff":
            # wavelength must be converted to mm for frequency units cycles/mm
            self.max_freq = 1 / (self.wavelength * 1e-3 * self.FNO)

        self.psf = [
            FFTPSF(
                self.optic,
                field,
                self.wavelength,
                self.num_rays,
                self.grid_size,
            ).psf
            for field in self.fields
        ]

        self.mtf = self._generate_mtf_data()

    def view(self, figsize=(12, 4), add_reference=False):
        """Visualizes the Modulation Transfer Function (MTF).

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (12, 4).
            add_reference (bool, optional): Whether to add the diffraction
                limit reference line. Defaults to False.

        """
        dx = self._get_mtf_units()
        freq = be.arange(self.grid_size // 2) * dx

        _, ax = plt.subplots(figsize=figsize)

        for k, data in enumerate(self.mtf):
            self._plot_field(ax, freq, data, self.fields[k], color=f"C{k}")

        if add_reference:
            ratio = freq / self.max_freq
            ratio = be.clip(ratio, -1, 1)  # avoid invalid value in arccos
            phi = be.arccos(ratio)
            diff_limited_mtf = 2 / be.pi * (phi - be.cos(phi) * be.sin(phi))
            ax.plot(freq, diff_limited_mtf, "k--", label="Diffraction Limit")

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim([0, self.max_freq])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Frequency (cycles/mm)", labelpad=10)
        ax.set_ylabel("Modulation Transfer Function", labelpad=10)
        plt.tight_layout()
        plt.show()

    def _plot_field(self, ax, freq, mtf_data, field, color):
        """Plot the MTF data for a specific field.

        Args:
            ax (matplotlib.axes.Axes): The axes object to plot on.
            freq (array-like): The frequency values.
            mtf_data (array-like): The MTF data for the field.
            field (tuple): The field values (Hx, Hy).
            color (str): The color of the plot.

        """
        ax.plot(
            freq,
            mtf_data[0],
            label=f"Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Tangential",
            color=color,
            linestyle="-",
        )
        ax.plot(
            freq,
            mtf_data[1],
            label=f"Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Sagittal",
            color=color,
            linestyle="--",
        )

    def _generate_mtf_data(self):
        """Generates the MTF (Modulation Transfer Function) data for each field.
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

    def _get_fno(self):
        """Calculate the effective F-number (FNO) of the optical system.
        Applies a correction if the object is finite.

        Returns:
            float: The effective F-number of the optical system.

        """
        FNO = self.optic.paraxial.FNO()

        if not self.optic.object_surface.is_infinite:
            D = self.optic.paraxial.XPD()
            p = D / self.optic.paraxial.EPD()
            m = self.optic.paraxial.magnification()
            FNO *= 1 + be.abs(m) / p

        return FNO

    def _get_mtf_units(self):
        """Calculate the MTF (Modulation Transfer Function) units.

        Returns:
            float: The MTF units calculated based on the grid size, number
                of rays, wavelength, and F-number.

        """
        Q = self.grid_size / self.num_rays
        dx = Q / (self.wavelength * self.FNO)

        return dx
