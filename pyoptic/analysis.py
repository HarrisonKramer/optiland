from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class SpotDiagram:

    def __init__(self, optic, fields='all', wavelengths='all', num_rays=100,
                 distribution='hexapolar'):
        self.optic = optic
        self.fields = fields
        self.wavelengths = wavelengths
        if self.fields == 'all':
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelengths == 'all':
            self.wavelengths = self.optic.wavelengths.get_wavelengths()

        self.data = self._generate_data(self.fields, self.wavelengths, num_rays, distribution)

    def view(self, figsize=(12, 4)):

        N = self.optic.fields.num_fields
        num_rows = (N + 2) // 3

        fig, axs = plt.subplots(num_rows, 3, figsize=(figsize[0], num_rows * figsize[1]))
        axs = axs.flatten()

        # subtract centroid and find limits
        data = self._center_spots(deepcopy(self.data))
        geometric_size = self.geometric_spot_radius()
        axis_lim = np.max(geometric_size)

        # plot wavelengths for each field
        for k, field_data in enumerate(data):
            self._plot_field(axs[k], field_data, self.fields[k], axis_lim, self.wavelengths)

        # remove empty axes
        for k in range(N, num_rows * 3):
            fig.delaxes(axs[k])

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

        plt.tight_layout()
        plt.show()

    def centroid(self):
        norm_index = self.optic.wavelengths.primary_index
        centroid = []
        for field_data in self.data:
            centroid_x = np.mean(field_data[norm_index][0])
            centroid_y = np.mean(field_data[norm_index][1])
            centroid.append((centroid_x, centroid_y))
        return centroid

    def geometric_spot_radius(self):
        data = self._center_spots(deepcopy(self.data))
        geometric_size = []
        for field_data in data:
            geometric_size_field = []
            for wave_data in field_data:
                r = np.sqrt(wave_data[0]**2 + wave_data[1]**2)
                geometric_size_field.append(np.max(r))
            geometric_size.append(geometric_size_field)
        return geometric_size

    def rms_spot_radius(self):
        data = self._center_spots(deepcopy(self.data))
        rms = []
        for field_data in data:
            rms_field = []
            for wave_data in field_data:
                r2 = wave_data[0]**2 + wave_data[1]**2
                rms_field.append(np.sqrt(np.mean(r2)))
            rms.append(rms_field)
        return rms

    def _center_spots(self, data):
        centroids = self.centroid()
        data = deepcopy(self.data)
        for i, field_data in enumerate(data):
            for wave_data in field_data:
                wave_data[0] -= centroids[i][0]
                wave_data[1] -= centroids[i][1]
        return data

    def _generate_data(self, fields, wavelengths, num_rays=100, distribution='hexapolar'):
        data = []
        for field in fields:
            field_data = []
            for wavelength in wavelengths:
                field_data.append(self._generate_field_data(field, wavelength, num_rays, distribution))
            data.append(field_data)
        return data

    def _generate_field_data(self, field, wavelength, num_rays=100, distribution='hexapolar'):
        self.optic.trace(*field, wavelength, num_rays, distribution)
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        return [x, y]

    def _plot_field(self, ax, field_data, field, axis_lim, wavelengths, buffer=1.05):
        markers = ['o', 's', '^']
        for k, points in enumerate(field_data):
            ax.scatter(*points, s=10, label=f'{wavelengths[k]:.4f} µm', marker=markers[k % 3])
            ax.axis('square')
            ax.set_xlabel('X (µm)')
            ax.set_ylabel('Y (µm)')
            ax.set_xlim((-axis_lim*buffer, axis_lim*buffer))
            ax.set_ylim((-axis_lim*buffer, axis_lim*buffer))
        ax.set_title(f'Hx: {field[0]:.3f}, Hy: {field[1]:.3f}')


class EncircledEnergy(SpotDiagram):

    def __init__(self, optic, fields='all', wavelength='primary', num_rays=100_000,
                 distribution='random', num_points=256):
        self.num_points = num_points
        if wavelength == 'primary':
            wavelength = optic.primary_wavelength

        super().__init__(optic, fields, [wavelength], num_rays, distribution)

        # self.ee, self.radius = 0

    def view(self):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        data = self._center_spots(deepcopy(self.data))
        geometric_size = self.geometric_spot_radius()
        axis_lim = np.max(geometric_size)
        for k, field_data in enumerate(data):
            self._plot_field(ax, field_data, self.fields[k], axis_lim, self.num_points)

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        ax.set_xlabel('Radius (mm)')
        ax.set_ylabel('Encircled Energy (-)')
        ax.set_title(f'Wavelength: {self.wavelengths[0]:.4f} µm')
        ax.set_xlim((0, None))
        ax.set_ylim((0, None))
        fig.tight_layout()
        plt.show()

    def centroid(self):
        centroid = []
        for field_data in self.data:
            centroid_x = np.mean(field_data[0][0])
            centroid_y = np.mean(field_data[0][1])
            centroid.append((centroid_x, centroid_y))
        return centroid

    def _encircled_energy(self, radii, energy, radius_max):
        return np.nansum(energy[radii <= radius_max])

    def _plot_field(self, ax, field_data, field, axis_lim, num_points, buffer=1.2):
        r_max = axis_lim * buffer
        r_step = np.linspace(0, r_max, num_points)
        for points in field_data:
            x, y, energy = points
            radii = np.sqrt(x**2 + y**2)
            def vectorized_ee(r): return np.nansum(energy[radii <= r])
            ee = np.vectorize(vectorized_ee)(r_step)
            ax.plot(r_step, ee, label=f'Hx: {field[0]:.3f}, Hy: {field[1]:.3f}')

    def _generate_encircled_energy_data(self, field_data, axis_lim, num_points, buffer=1.2):
        # TODO - complete
        r_max = axis_lim * buffer
        r_step = np.linspace(0, r_max, num_points)
        ee = []
        for points in field_data:
            x, y, energy = points
            radii = np.sqrt(x**2 + y**2)
            def vectorized_ee(r): return np.nansum(energy[radii <= r])
            ee.append(np.vectorize(vectorized_ee)(r_step))
        return r_step, ee

    def _generate_field_data(self, field, wavelength, num_rays=100, distribution='hexapolar'):
        self.optic.trace(*field, wavelength, num_rays, distribution)
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        energy = self.optic.surface_group.energy[-1, :]
        return [x, y, energy]


class RayFan:

    def __init__(self, optic, fields='all', wavelengths='all', num_points=256):
        self.optic = optic
        self.fields = fields
        self.wavelengths = wavelengths
        if num_points % 2 == 1:
            num_points += 1  # force to be odd so a point lies at P=0
        self.num_points = num_points

        if self.fields == 'all':
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelengths == 'all':
            self.wavelengths = self.optic.wavelengths.get_wavelengths()

        self.data = self._generate_data()

    def view(self):
        _, axs = plt.subplots(nrows=len(self.fields), ncols=2, figsize=(10, 10), sharex=True, sharey=True)

        Px = self.data['Px']
        Py = self.data['Py']

        for k, field in enumerate(self.fields):
            for wavelength in self.wavelengths:
                ex = self.data[f'{field}'][f'{wavelength}']['x']
                ey = self.data[f'{field}'][f'{wavelength}']['y']

                axs[k, 0].plot(Py, ey, zorder=3, label=f'{wavelength:.4f} µm')
                axs[k, 0].grid()
                axs[k, 0].axhline(y=0, lw=1, color='gray')
                axs[k, 0].axvline(x=0, lw=1, color='gray')
                axs[k, 0].set_xlabel('$P_y$')
                axs[k, 0].set_ylabel('$\\epsilon_y$ (mm)')
                axs[k, 0].set_xlim((-1, 1))
                axs[k, 0].set_title(f'Hx: {field[0]:.3f}, Hy: {field[1]:.3f}')

                axs[k, 1].plot(Px, ex, zorder=3, label=f'{wavelength:.4f} µm')
                axs[k, 1].grid()
                axs[k, 1].axhline(y=0, lw=1, color='gray')
                axs[k, 1].axvline(x=0, lw=1, color='gray')
                axs[k, 1].set_xlabel('$P_x$')
                axs[k, 1].set_ylabel('$\\epsilon_x$ (mm)')
                axs[k, 0].set_xlim((-1, 1))
                axs[k, 1].set_title(f'Hx: {field[0]:.3f}, Hy: {field[1]:.3f}')

        plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2), ncol=3)
        plt.subplots_adjust(top=1)
        plt.show()

    def _generate_data(self):
        data = {}
        data['Px'] = np.linspace(-1, 1, self.num_points)
        data['Py'] = np.linspace(-1, 1, self.num_points)
        for field in self.fields:
            Hx = field[0]
            Hy = field[1]

            data[f'{field}'] = {}
            for wavelength in self.wavelengths:
                data[f'{field}'][f'{wavelength}'] = {}

                self.optic.trace(Hx=Hx, Hy=Hy, wavelength=wavelength,
                                 num_rays=self.num_points, distribution='line_x')
                data[f'{field}'][f'{wavelength}']['x'] = self.optic.surface_group.x[-1, :]

                self.optic.trace(Hx=Hx, Hy=Hy, wavelength=wavelength,
                                 num_rays=self.num_points, distribution='line_y')
                data[f'{field}'][f'{wavelength}']['y'] = self.optic.surface_group.y[-1, :]

        # remove distortion
        wave_ref = self.optic.primary_wavelength
        for field in self.fields:
            x_offset = data[f'{field}'][f'{wave_ref}']['x'][self.num_points//2]
            y_offset = data[f'{field}'][f'{wave_ref}']['y'][self.num_points//2]
            for wavelength in self.wavelengths:
                data[f'{field}'][f'{wavelength}']['x'] -= x_offset
                data[f'{field}'][f'{wavelength}']['y'] -= y_offset

        return data


class GeometricMTF(SpotDiagram):
    """Smith, Modern Optical Engineering 3rd edition, Section 11.9"""

    def __init__(self, optic, fields='all', wavelength='primary', num_rays=100_000,
                 distribution='square', num_points=256, max_freq='cutoff', scale=True):
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

        mtf = []  # TODO: add option more polychromatic MTF
        for field_data in self.data:
            xi, yi = field_data[0][0], field_data[0][1]
            mtf.append([self._compute_geometric_mtf(yi, self.freq, scale_factor),
                        self._compute_geometric_mtf(xi, self.freq, scale_factor)])
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


class YYbar:

    def __init__(self, optic, wavelength='primary'):
        self.optic = optic
        if wavelength == 'primary':
            wavelength = optic.primary_wavelength
        self.wavelength = wavelength

    def view(self):
        ya, _ = self.optic.paraxial.marginal_ray()
        yb, _ = self.optic.paraxial.chief_ray()

        plt.plot(yb, ya)
        plt.axis([-25, 25, -25, 25])
        # plt.axis('image')
        plt.show()


# TODO: OPD fans
# TODO: distortion plot
# TODO: grid distortion
# TODO: field curvature
