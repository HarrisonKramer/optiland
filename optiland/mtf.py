import numpy as np
import matplotlib.pyplot as plt
from optiland.analysis import SpotDiagram
from optiland.psf import FFTPSF


class GeometricMTF(SpotDiagram):
    """Smith, Modern Optical Engineering 3rd edition, Section 11.9"""

    def __init__(self, optic, fields='all', wavelength='primary',
                 num_rays=100, distribution='uniform', num_points=256,
                 max_freq='cutoff', scale=True):
        self.num_points = num_points
        self.scale = scale

        if wavelength == 'primary':
            wavelength = optic.primary_wavelength
        if max_freq == 'cutoff':
            # wavelength must be converted to mm for frequency units cycles/mm
            self.max_freq = 1 / (wavelength * 1e-3 * optic.paraxial.FNO())

        super().__init__(optic, fields, [wavelength], num_rays, distribution)

        self.freq = np.linspace(0, self.max_freq, num_points)
        self.mtf = self._generate_mtf_data()

    def view(self, figsize=(12, 4)):
        _, ax = plt.subplots(figsize=figsize)

        for k, data in enumerate(self.mtf):
            self._plot_field(ax, data, self.fields[k], color=f'C{k}')

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        ax.set_xlim([0, self.max_freq])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Frequency (cycles/mm)', labelpad=10)
        ax.set_ylabel('Modulation', labelpad=10)
        plt.tight_layout()
        plt.show()

    def _generate_mtf_data(self):
        if self.scale:
            phi = np.arccos(self.freq / self.max_freq)
            scale_factor = 2 / np.pi * (phi - np.cos(phi) * np.sin(phi))
        else:
            scale_factor = 1

        mtf = []  # TODO: add option for polychromatic MTF
        for field_data in self.data:
            xi, yi = field_data[0][0], field_data[0][1]
            mtf.append([self._compute_field_data(yi, self.freq, scale_factor),
                        self._compute_field_data(xi, self.freq, scale_factor)])
        return mtf

    def _compute_field_data(self, xi, v, scale_factor):
        A, edges = np.histogram(xi, bins=self.num_points+1)
        x = (edges[1:] + edges[:-1]) / 2
        dx = x[1] - x[0]

        mtf = np.zeros_like(v)
        for k in range(len(v)):
            Ac = np.sum(A * np.cos(2 * np.pi * v[k] * x) * dx) / np.sum(A * dx)
            As = np.sum(A * np.sin(2 * np.pi * v[k] * x) * dx) / np.sum(A * dx)

            mtf[k] = np.sqrt(Ac**2 + As**2)

        return mtf * scale_factor

    def _plot_field(self, ax, mtf_data, field, color):
        ax.plot(self.freq, mtf_data[0],
                label=f'Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Tangential',
                color=color, linestyle='-')
        ax.plot(self.freq, mtf_data[1],
                label=f'Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Sagittal',
                color=color, linestyle='--')


class FFTMTF:
    # TODO: verify performance against baseline

    def __init__(self, optic, fields='all', wavelength='primary',
                 num_rays=128, grid_size=1024, max_freq='cutoff'):
        self.optic = optic
        self.max_freq = max_freq
        self.fields = fields
        self.wavelength = wavelength
        self.num_rays = num_rays
        self.grid_size = grid_size

        if self.fields == 'all':
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelength == 'primary':
            self.wavelength = optic.primary_wavelength

        if max_freq == 'cutoff':
            # wavelength must be converted to mm for frequency units cycles/mm
            self.max_freq = 1 / (self.wavelength * 1e-3 * optic.paraxial.FNO())

        self.psf = [FFTPSF(self.optic, field, self.wavelength,
                           self.num_rays, self.grid_size).psf
                    for field in self.fields]

        self.mtf = self._generate_mtf_data()

    def view(self, figsize=(12, 4)):
        dx = self._get_mtf_units()
        freq = np.arange(self.grid_size//2) * dx

        _, ax = plt.subplots(figsize=figsize)

        for k, data in enumerate(self.mtf):
            self._plot_field(ax, freq, data, self.fields[k], color=f'C{k}')

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        ax.set_xlim([0, self.max_freq])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Frequency (cycles/mm)', labelpad=10)
        ax.set_ylabel('Modulation Transfer Function', labelpad=10)
        plt.tight_layout()
        plt.show()

    def _plot_field(self, ax, freq, mtf_data, field, color):
        ax.plot(freq, mtf_data[0],
                label=f'Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Tangential',
                color=color, linestyle='-')
        ax.plot(freq, mtf_data[1],
                label=f'Hx: {field[0]:.1f}, Hy: {field[1]:.1f}, Sagittal',
                color=color, linestyle='--')

    def _generate_mtf_data(self):
        mtf_data = [np.abs(np.fft.fftshift(np.fft.fft2(psf)))
                    for psf in self.psf]
        mtf = []
        for data in mtf_data:
            tangential = data[self.grid_size//2:, self.grid_size//2]
            sagittal = data[self.grid_size//2, self.grid_size//2:]
            mtf.append([tangential/np.max(tangential),
                        sagittal/np.max(sagittal)])
        return mtf

    def _get_mtf_units(self):
        FNO = self.optic.paraxial.FNO()

        if not self.optic.object_surface.is_infinite:
            D = self.optic.paraxial.XPD()
            p = D / self.optic.paraxial.EPD()
            m = self.optic.paraxial.magnification()
            FNO *= (1 + np.abs(m) / p)

        Q = self.grid_size / self.num_rays
        dx = Q / (self.wavelength * FNO)

        return dx