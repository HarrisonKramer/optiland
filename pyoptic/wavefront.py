import numpy as np
import matplotlib.pyplot as plt
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

    def __init__(self, optic, fields='all', wavelengths='all', num_rays=256):
        self.pupil_coord = np.linspace(-1, 1, num_rays)
        super().__init__(optic, fields=fields, wavelengths=wavelengths,
                         num_rays=num_rays, distribution='uniform')

    def view(self):
        pass


class FFTPSF(Wavefront):

    def __init__(self, optic, field, wavelengths='all',
                 num_rays=128, grid_size=1024):
        super().__init__(optic=optic, fields=[field], wavelengths=wavelengths,
                         num_rays=num_rays, distribution='uniform')

        self.grid_size = grid_size
        self.psf = self._compute_psf()

    def view(self, projection='2d'):
        if projection == '2d':
            self._plot_2d()
        elif projection == '3d':
            self._plot_3d()
        else:
            raise ValueError('OPD projection must be "2d" or "3d".')

    def _plot_2d(self):
        pass

    def _plot_3d(self):
        pass

    def _compute_psf(self):
        pupils = self._generate_pupils()
        norm_factor = self._get_normalization(pupils)

        psf = []
        for pupil in pupils:
            amp = np.fft.fftshift(np.fft.fft2(pupil))
            psf.append(amp * np.conj(amp))

        return np.real(np.sum(psf, axis=0)) / norm_factor * 100

    def _generate_pupils(self):
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

            pad = (self.grid_size - P.shape[0]) // 2
            P = np.pad(P, ((pad, pad), (pad, pad)),
                       mode='constant', constant_values=0)

            pupils.append(P)

        return pupils

    def _get_normalization(self, pupils):
        P_nom = pupils[0].copy()
        P_nom[P_nom != 0] = 1

        amp_norm = np.fft.fftshift(np.fft.fft2(P_nom))
        psf_norm = amp_norm * np.conj(amp_norm)
        return np.real(np.max(psf_norm) * len(pupils))

    def _get_psf_units(self):
        D = self.optic.paraxial.XPD()
        wavelength = 0.6563e-3
        Q = self.grid_size / self.num_rays
