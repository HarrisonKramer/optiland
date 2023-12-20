import numpy as np
import matplotlib.pyplot as plt
from optiland.analysis import SpotDiagram
from optiland.psf import FFTPSF


class GeometricMTF(SpotDiagram):
    """Smith, Modern Optical Engineering 3rd edition, Section 11.9"""

    def __init__(self, optic, fields='all', wavelength='primary',
                 num_rays=100_000, distribution='square', num_points=256,
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
            mtf.append([self._compute_geometric_mtf(yi, self.freq,
                                                    scale_factor),
                        self._compute_geometric_mtf(xi, self.freq,
                                                    scale_factor)])
        return mtf

    def _compute_geometric_mtf(self, xi, v, scale_factor):
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


class FFTMTF(FFTPSF):

    def __init__(self, optic, field, wavelengths='all',
                 num_rays=128, grid_size=1024):
        super().__init__(optic, field, wavelengths, num_rays, grid_size)

    def view(self):
        pass

    def _compute_mtf(self):
        pass

    def _get_mtf_units(self):
        pass

    def _plot_2d(self):
        """Override to disable function"""
        pass

    def _plot_3d(self):
        """Override to disable function"""
        pass
