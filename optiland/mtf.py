"""Modulation Transfer Function (MTF) Module

This module provides various classes for the computation of the modulation
transfer function (MTF) of an optical system.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.analysis import SpotDiagram
from optiland.psf import FFTPSF
from optiland.wavefront import Wavefront
from optiland.zernike import ZernikeFit


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
        freq (be.ndarray): The frequency values for the MTF curve.
        mtf (list): The MTF data for each field point. Each element is a list
            containing tangential and sagittal MTF data (`be.ndarray`) for a field.
        diff_limited_mtf (be.ndarray): The diffraction-limited MTF curve.

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
            ax.plot(
                be.to_numpy(self.freq),
                be.to_numpy(self.diff_limited_mtf),
                "k--",
                label="Diffraction Limit",
            )

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim([0, be.to_numpy(self.max_freq)])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Frequency (cycles/mm)", labelpad=10)
        ax.set_ylabel("Modulation", labelpad=10)
        plt.tight_layout()
        plt.grid(alpha=0.25)
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
            spot_data_item = field_data[0]
            xi, yi = spot_data_item.x, spot_data_item.y
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
            xi (be.ndarray): The coordinate values (x or y) of the field point.
            v (be.ndarray): The frequency values for the MTF curve.
            scale_factor (float or be.ndarray): The scale factor for the MTF curve.

        Returns:
            be.ndarray: The MTF data for the field point.

        """
        A, edges = be.histogram(xi, bins=self.num_points + 1)
        x = (edges[1:] + edges[:-1]) / 2
        dx = x[1] - x[0]

        mtf = be.copy(be.zeros_like(v))  # copy required to maintain gradient
        for k in range(len(v)):
            Ac = be.sum(A * be.cos(2 * be.pi * v[k] * x) * dx) / be.sum(A * dx)
            As = be.sum(A * be.sin(2 * be.pi * v[k] * x) * dx) / be.sum(A * dx)

            mtf[k] = be.sqrt(Ac**2 + As**2)

        return mtf * scale_factor

    def _plot_field(self, ax, mtf_data, field, color):
        """Plots the MTF data for a given field point.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            mtf_data (list[be.ndarray]): The MTF data for the field point,
                containing tangential and sagittal MTF arrays.
            field (tuple[float, float]): The field point coordinates (Hx, Hy).
            color (str): The color of the plotted lines.

        """
        ax.plot(
            be.to_numpy(self.freq),
            be.to_numpy(mtf_data[0]),
            label=f"Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Tangential",
            color=color,
            linestyle="-",
        )
        ax.plot(
            be.to_numpy(self.freq),
            be.to_numpy(mtf_data[1]),
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
            ax.plot(
                be.to_numpy(freq),
                be.to_numpy(diff_limited_mtf),
                "k--",
                label="Diffraction Limit",
            )

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim([0, be.to_numpy(self.max_freq)])
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
            be.to_numpy(freq),
            be.to_numpy(mtf_data[0]),
            label=f"Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Tangential",
            color=color,
            linestyle="-",
        )
        ax.plot(
            be.to_numpy(freq),
            be.to_numpy(mtf_data[1]),
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
            FNO = FNO * (1 + be.abs(m) / p)

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


class SampledMTF:
    """Sampled Modulation Transfer Function (MTF) class.

    This class calculates the MTF of an optical system from sampled wavefront
    data. It utilizes Zernike polynomial fitting to represent the wavefront
    aberrations.

    Note:
        This class assumes that amplitude variations between the pupil and a
        shifted version of the pupil can be ignored.

    Args:
        optic (Optic): The optical system.
        field (tuple): The field point (Hx, Hy) at which to calculate the MTF.
        wavelength (str or float): The wavelength (in mm) at which to
            calculate the MTF. Can be 'primary' to use the optic's primary
            wavelength.
        num_rays (int, optional): The number of rings to trace for the
            wavefront analysis. Defaults to 128 in each axis.
        distribution (str, optional): The distribution of rays in the pupil.
            Defaults to 'uniform'.
        zernike_terms (int, optional): The number of Zernike terms to use for
            the wavefront fit. Defaults to 37.
        zernike_type (str, optional): The type of Zernike polynomials to use
            ('fringe', 'standard', etc.). Defaults to 'fringe'.

    Attributes:
        optic (Optic): The optical system.
        field (tuple): The field point (Hx, Hy).
        wavelength (float): The wavelength (in mm) used for calculation.
        num_rays (int): The number of rays used for wavefront analysis.
        distribution (str): The ray distribution in the pupil.
        zernike_terms (int): The number of Zernike terms for the fit.
        zernike_type (str): The type of Zernike polynomials used.
        x_norm (be.ndarray): Normalized x-coordinates of pupil samples.
        y_norm (be.ndarray): Normalized y-coordinates of pupil samples.
        opd_waves (be.ndarray): Optical Path Difference (OPD) in waves.
        intensity (be.ndarray): Intensity at each pupil sample point.
        zernike_fit (ZernikeFit): The Zernike fit object.
        P1 (be.ndarray): The complex pupil function.
        otf_at_zero (float): The value of the Optical Transfer Function (OTF)
            at zero frequency, equivalent to the sum of intensities.
    """

    def __init__(
        self,
        optic,
        field,
        wavelength,
        num_rays=128,
        distribution="uniform",
        zernike_terms=37,
        zernike_type="fringe",
    ):
        """Initializes the SampledMTF instance."""
        self.optic = optic
        self.field = field
        self.wavelength = wavelength
        self.num_rays = num_rays
        self.distribution = distribution
        self.zernike_terms = zernike_terms
        self.zernike_type = zernike_type

        if self.wavelength == "primary":
            self.wavelength = optic.primary_wavelength

        wf = Wavefront(
            optic,
            fields=[field],
            wavelengths=[self.wavelength],
            num_rays=num_rays,
            distribution=distribution,
        )
        wf_data = wf.get_data(field, self.wavelength)

        self.x_norm = wf.distribution.x
        self.y_norm = wf.distribution.y
        self.opd_waves = wf_data.opd
        self.intensity = wf_data.intensity

        self.xpd = self.optic.paraxial.XPD()
        self.xpl = -self.optic.paraxial.XPL()

        self.zernike_fit = ZernikeFit(
            self.x_norm,
            self.y_norm,
            self.opd_waves,
            self.zernike_type,
            self.zernike_terms,
        )

        self.P1 = be.sqrt(self.intensity) * be.exp(1j * 2 * be.pi * self.opd_waves)
        self.otf_at_zero = be.sum(self.intensity)

    def calculate_mtf(self, frequencies):
        """Calculates the Modulation Transfer Function (MTF) for given spatial
        frequencies.

        The method computes the MTF by determining the Optical Transfer Function (OTF)
        from the overlap integral of the pupil function with a shifted version of
        itself. The shift corresponds to the spatial frequency being evaluated.
        The MTF is the absolute value of the normalized OTF.

        Args:
            frequencies (list[tuple[float, float]] or be.ndarray): A list or array of
                tuples, where each tuple `(fx, fy)` represents a spatial frequency
                pair in cycles per mm for which to calculate the MTF.

        Returns:
            list[float]: A list of MTF values corresponding to each input frequency
            pair. The MTF values are dimensionless and range from 0 to 1.

        Notes:
            The calculation involves:
            1. Retrieving the exit pupil diameter (XPD). If XPD is near zero,
               MTF is 0 for non-zero frequencies and 1 for zero frequency.
            2. Converting the wavelength to mm.
            3. For each frequency pair (fx, fy):
                a. Calculating physical shifts in the pupil based on wavelength and
                   frequency.
                b. Normalizing these shifts using the XPD radius.
                c. Determining the shifted normalized coordinates for pupil evaluation.
                d. Evaluating the Optical Path Difference (OPD) at these shifted
                   coordinates using the Zernike polynomial fit of the wavefront.
                e. Masking points where the shifted evaluation falls outside the
                   unit circle (i.e., outside the pupil).
                f. Computing the complex conjugate of the pupil function at the
                   shifted coordinates (P2_conj).
                g. Calculating the OTF element as the product of the original complex
                   pupil function (P1) and P2_conj.
                h. Summing the OTF elements over all pupil sample points to get the
                   total OTF value for the given frequency.
                i. Normalizing the OTF value by otf_at_zero (the OTF at zero frequency).
                j. The MTF is the absolute value of this normalized OTF.
        """
        xpd_is_zero = be.isclose(self.xpd, 0.0, atol=1e-9)

        # Retrieve necessary attributes from instance
        wl_um = self.wavelength  # Wavelength in micrometers
        x_norm = self.x_norm
        y_norm = self.y_norm
        intensity = self.intensity
        P1 = self.P1
        zernike_fit = self.zernike_fit

        wl_mm = wl_um * 1e-3  # Convert wavelength to mm

        mtf_results = []

        for fx, fy in frequencies:
            if xpd_is_zero:
                if be.isclose(fx, 0.0) and be.isclose(fy, 0.0):
                    mtf_results.append(1.0)
                else:
                    mtf_results.append(0.0)
                continue

            delta_x_phys = wl_mm * fx
            delta_y_phys = wl_mm * fy

            delta_x_norm = self.xpl * delta_x_phys / (self.xpd / 2)
            delta_y_norm = self.xpl * delta_y_phys / (self.xpd / 2)

            eval_x_shifted_norm = x_norm - delta_x_norm
            eval_y_shifted_norm = y_norm - delta_y_norm

            r_shifted_norm = be.sqrt(eval_x_shifted_norm**2 + eval_y_shifted_norm**2)
            phi_shifted_norm = be.arctan2(eval_y_shifted_norm, eval_x_shifted_norm)

            opd_shifted_waves = zernike_fit.zernike.poly(
                r_shifted_norm, phi_shifted_norm
            )

            mask_outside_pupil = r_shifted_norm > 1.0

            # Complex conjugate of the shifted pupil function P2
            # Intensity is assumed to be the same for P2 as for P1 at corresponding
            # original points
            P2_conj = be.sqrt(intensity) * be.exp(-1j * 2 * be.pi * opd_shifted_waves)
            P2_conj = be.where(mask_outside_pupil, 0.0 + 0.0j, P2_conj)

            otf_element = P1 * P2_conj
            otf_value = be.sum(otf_element)

            normalized_otf = otf_value / self.otf_at_zero
            mtf = be.abs(normalized_otf)
            mtf_results.append(mtf)

        return mtf_results
