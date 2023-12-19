import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from scipy.interpolate import griddata
from pyoptic.distribution import create_distribution


class Wavefront:

    def __init__(self, optic, fields='all', wavelengths='all', num_rays=100,
                 distribution='hexapolar'):
        self.optic = optic
        self.fields = fields
        self.wavelengths = wavelengths
        self.num_rays = num_rays

        if self.fields == 'all':
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelengths == 'all':
            self.wavelengths = self.optic.wavelengths.get_wavelengths()
        elif self.wavelengths == 'primary':
            self.wavelengths = [optic.primary_wavelength]

        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        self.distribution = distribution

        self.data = self._generate_data(self.fields, self.wavelengths)

    def _generate_data(self, fields, wavelengths):
        pupil_z = (self.optic.paraxial.XPL() +
                   self.optic.surface_group.positions[-1])

        data = []
        for field in fields:
            field_data = []
            for wavelength in wavelengths:
                # Trace chief ray for field & find reference sphere properties
                self._trace_chief_ray(field, wavelength)

                # Reference sphere center and radius
                xc, yc, zc, R = self._get_reference_sphere(pupil_z)
                opd_ref = self._get_path_length(xc, yc, zc, R)
                opd_ref = self._correct_tilt(field, opd_ref, x=0, y=0)

                field_data.append(self._generate_field_data(field, wavelength,
                                                            opd_ref,
                                                            xc, yc, zc, R))
            data.append(field_data)
        return data

    def _generate_field_data(self, field, wavelength, opd_ref, xc, yc, zc, R):
        # trace distribution through pupil
        self.optic.trace(*field, wavelength, None, self.distribution)
        opd = self._get_path_length(xc, yc, zc, R)
        opd = self._correct_tilt(field, opd)
        return (opd_ref - opd) / (wavelength * 1e-3)

    def _trace_chief_ray(self, field, wavelength):
        self.optic.trace_generic(*field, Px=0.0, Py=0.0, wavelength=wavelength)

    def _get_reference_sphere(self, pupil_z):
        if self.optic.surface_group.x[-1, :].size != 1:
            raise ValueError('Chief ray cannot be determined. '
                             'It must be traced alone.')

        # chief ray intersection location
        xc = self.optic.surface_group.x[-1, :]
        yc = self.optic.surface_group.y[-1, :]
        zc = self.optic.surface_group.z[-1, :]

        # radius of sphere - exit pupil origin vs. center
        R = np.sqrt(xc**2 + yc**2 + (zc - pupil_z)**2)

        return xc, yc, zc, R

    def _get_path_length(self, xc, yc, zc, r):
        opd = self.optic.surface_group.opd[-1, :]
        return opd - self._opd_image_to_xp(xc, yc, zc, r)

    def _correct_tilt(self, field, opd, x=None, y=None):
        tilt_correction = 0
        if self.optic.field_type == 'angle':
            Hx, Hy = field
            x_tilt = self.optic.fields.max_x_field * Hx
            y_tilt = self.optic.fields.max_y_field * Hy
            if x is None:
                x = self.distribution.x
            if y is None:
                y = self.distribution.y
            EPD = self.optic.paraxial.EPD()
            tilt_correction = ((1 - x) * np.sin(np.radians(x_tilt)) * EPD / 2 +
                               (1 - y) * np.sin(np.radians(y_tilt)) * EPD / 2)
        return opd - tilt_correction

    def _opd_image_to_xp(self, xc, yc, zc, R):
        xr = self.optic.surface_group.x[-1, :]
        yr = self.optic.surface_group.y[-1, :]
        zr = self.optic.surface_group.z[-1, :]

        L = -self.optic.surface_group.L[-1, :]
        M = -self.optic.surface_group.M[-1, :]
        N = -self.optic.surface_group.N[-1, :]

        a = L**2 + M**2 + N**2
        b = 2*L*(xr - xc) + 2*M*(yr - yc) + 2*N*(zr - zc)
        c = (xr**2 + yr**2 + zr**2 - 2*xr*xc + xc**2 - 2*yr*yc + yc**2 -
             2*zr*zc + zc**2 - R**2)

        d = b ** 2 - 4 * a * c
        t = (-b - np.sqrt(d)) / (2 * a)
        try:
            t[t < 0] = (-b[t < 0] + np.sqrt(d[t < 0])) / (2 * a[t < 0])
        except TypeError:  # input is not an array
            if t < 0:
                t = (-b + np.sqrt(d)) / (2 * a)
        return t


class OPDFan(Wavefront):

    def __init__(self, optic, fields='all', wavelengths='all', num_rays=100):
        self.pupil_coord = np.linspace(-1, 1, num_rays)
        super().__init__(optic, fields=fields, wavelengths=wavelengths,
                         num_rays=num_rays, distribution='cross')

    def view(self, figsize=(10, 7)):
        num_rows = len(self.fields)
        _, axs = plt.subplots(nrows=len(self.fields), ncols=2,
                              figsize=(10, num_rows*3), sharex=True,
                              sharey=True)

        for i, field in enumerate(self.fields):
            for j, wavelength in enumerate(self.wavelengths):
                wx = self.data[i][j][self.num_rays:]
                wy = self.data[i][j][:self.num_rays]

                axs[i, 0].plot(self.pupil_coord, wy, zorder=3,
                               label=f'{wavelength:.4f} µm')
                axs[i, 0].grid()
                axs[i, 0].axhline(y=0, lw=1, color='gray')
                axs[i, 0].axvline(x=0, lw=1, color='gray')
                axs[i, 0].set_xlabel('$P_y$')
                axs[i, 0].set_ylabel('Wavefront Error (waves)')
                axs[i, 0].set_xlim((-1, 1))
                axs[i, 0].set_title(f'Hx: {field[0]:.3f}, Hy: {field[1]:.3f}')

                axs[i, 1].plot(self.pupil_coord, wx, zorder=3,
                               label=f'{wavelength:.4f} µm')
                axs[i, 1].grid()
                axs[i, 1].axhline(y=0, lw=1, color='gray')
                axs[i, 1].axvline(x=0, lw=1, color='gray')
                axs[i, 1].set_xlabel('$P_x$')
                axs[i, 1].set_ylabel('Wavefront Error (waves)')
                axs[i, 0].set_xlim((-1, 1))
                axs[i, 1].set_title(f'Hx: {field[0]:.3f}, Hy: {field[1]:.3f}')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.subplots_adjust(top=1)
        plt.tight_layout()
        plt.show()


class OPD(Wavefront):

    def __init__(self, optic, fields='all', wavelengths='all',
                 num_rays=256, grid_size=1024):
        self.pupil_coord = np.linspace(-1, 1, num_rays)
        self.grid_size = grid_size
        super().__init__(optic, fields=fields, wavelengths=wavelengths,
                         num_rays=num_rays, distribution='uniform')

        self.pupils = self._generate_pupils()

    def view(self):
        pass

    def _generate_pupils(self):
        # TODO: make more modular for PSF, don't pad here
        x = np.linspace(-1, 1, self.num_rays)
        x, y = np.meshgrid(x, x)
        x = x.ravel()
        y = y.ravel()
        R = np.sqrt(x**2 + y**2)

        pupils = []

        for k in range(len(self.wavelengths)):
            P = np.zeros_like(x, dtype=complex)
            P[R <= 1] = np.exp(1j * 2 * np.pi * self.data[0][k])
            P = np.reshape(P, (self.num_rays, self.num_rays))
            pupils.append(P)

        return pupils


class FFTPSF(OPD):
    # TODO: add transmission from object to exit pupil

    def __init__(self, optic, field, wavelengths='all',
                 num_rays=128, grid_size=1024):
        super().__init__(optic=optic, fields=[field], wavelengths=wavelengths,
                         num_rays=num_rays)

        self.grid_size = grid_size
        self.psf = self._compute_psf()

    def view(self, projection='2d', log=False, figsize=(6, 5), threshold=0.25):
        min_x, min_y, max_x, max_y = self._find_bounds(threshold)
        psf_zoomed = self.psf[min_x:max_x, min_y:max_y]
        psf_smooth = self._interpolate_psf(psf_zoomed)

        if projection == '2d':
            self._plot_2d(psf_smooth, log, figsize=(6, 5))
        elif projection == '3d':
            self._plot_3d(psf_smooth, log, figsize=(6, 5))
        else:
            raise ValueError('OPD projection must be "2d" or "3d".')

    def _plot_2d(self, image, log, figsize=(6, 5)):
        _, ax = plt.subplots(figsize=figsize)
        if log:
            norm = LogNorm()
        else:
            norm = None

        im = ax.imshow(image, norm=norm)

        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_title('FFT PSF')

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Relative Intensity (%)', rotation=270)
        plt.show()

    def _plot_3d(self, image, log, figsize=(6, 5)):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x = np.linspace(0, 1, image.shape[1])
        y = np.linspace(0, 1, image.shape[0])
        X, Y = np.meshgrid(x, y)

        if log:
            image = np.log10(image)
            ax.zaxis.set_major_formatter(mticker.FuncFormatter(self._log_tick_formatter))
            ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        surf = ax.plot_surface(X, Y, image, rstride=1, cstride=1,
                               cmap='viridis', linewidth=0, antialiased=False)

        ax.set_xlabel('X (µm)')
        ax.set_ylabel('X (µm)')
        ax.set_zlabel('Relative Intensity (%)')
        ax.set_title('FFT PSF')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        fig.tight_layout()
        plt.show()

    def _log_tick_formatter(self, val, pos=None):
        """https://stackoverflow.com/questions/3909794/plotting-mplot3d-axes3d-xyz-surface-plot-with-log-scale"""
        return f"$10^{{{int(val)}}}$"

    def _compute_psf(self):
        pupils = self._pad_pupils()
        norm_factor = self._get_normalization()

        psf = []
        for pupil in pupils:
            amp = np.fft.fftshift(np.fft.fft2(pupil))
            psf.append(amp * np.conj(amp))

        return np.real(np.sum(psf, axis=0)) / norm_factor * 100

    def _interpolate_psf(self, image, n=128):
        x_orig, y_orig = np.meshgrid(np.linspace(0, 1, image.shape[0]),
                                     np.linspace(0, 1, image.shape[1]))

        x_interp, y_interp = np.meshgrid(np.linspace(0, 1, n),
                                         np.linspace(0, 1, n))

        points = np.column_stack((x_orig.flatten(), y_orig.flatten()))
        values = image.flatten()

        return griddata(points, values, (x_interp, y_interp), method='cubic')

    def _find_bounds(self, threshold=0.25):
        thresholded_psf = self.psf > threshold
        non_zero_indices = np.argwhere(thresholded_psf)

        min_x, min_y = np.min(non_zero_indices, axis=0)
        max_x, max_y = np.max(non_zero_indices, axis=0)
        size = max(max_x - min_x, max_y - min_y)

        peak_x, peak_y = np.unravel_index(np.argmax(self.psf), self.psf.shape)

        min_x = peak_x - size / 2
        max_x = peak_x + size / 2
        min_y = peak_y - size / 2
        max_y = peak_y + size / 2

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(self.psf.shape[0], max_x)
        max_y = min(self.psf.shape[1], max_y)

        return int(min_x), int(min_y), int(max_x), int(max_y)

    def _pad_pupils(self):
        pupils_padded = []
        for pupil in self.pupils:
            pad = (self.grid_size - pupil.shape[0]) // 2
            pupil = np.pad(pupil, ((pad, pad), (pad, pad)),
                           mode='constant', constant_values=0)
            pupils_padded.append(pupil)
        return pupils_padded

    def _get_normalization(self):
        P_nom = self.pupils[0].copy()
        P_nom[P_nom != 0] = 1

        amp_norm = np.fft.fftshift(np.fft.fft2(P_nom))
        psf_norm = amp_norm * np.conj(amp_norm)
        return np.real(np.max(psf_norm) * len(self.pupils))

    def _get_psf_units(self):
        D = self.optic.paraxial.XPD()
        wavelength = 0.6563e-3
        Q = self.grid_size / self.num_rays


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
