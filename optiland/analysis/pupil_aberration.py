import numpy as np
import matplotlib.pyplot as plt


class PupilAberration:
    """
    Represents the pupil aberrations of an optic.

    Args:
        optic (Optic): The optic object to analyze.
        fields (str or list, optional): The fields to analyze.
            Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points in the pupil
            aberration. Defaults to 256.
    """

    def __init__(self, optic, fields='all', wavelengths='all', num_points=256):
        self.optic = optic
        self.fields = fields
        self.wavelengths = wavelengths
        if num_points % 2 == 0:
            num_points += 1  # force to be odd so a point lies at P=0
        self.num_points = num_points

        if self.fields == 'all':
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelengths == 'all':
            self.wavelengths = self.optic.wavelengths.get_wavelengths()

        self.data = self._generate_data()

    def view(self, figsize=(10, 3.33)):
        """
        Displays the pupil aberration plot.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (10, 3.33).
        """
        _, axs = plt.subplots(nrows=len(self.fields), ncols=2,
                              figsize=(figsize[0],
                                       figsize[1]*len(self.fields)),
                              sharex=True, sharey=True)

        # Ensure axs is a 2D array
        axs = np.atleast_2d(axs)

        Px = self.data['Px']
        Py = self.data['Py']

        for k, field in enumerate(self.fields):
            for wavelength in self.wavelengths:
                ex = self.data[f'{field}'][f'{wavelength}']['x']
                i_x = self.data[f'{field}'][f'{wavelength}']['intensity_x']
                ex[i_x == 0] = np.nan

                ey = self.data[f'{field}'][f'{wavelength}']['y']
                i_y = self.data[f'{field}'][f'{wavelength}']['intensity_y']
                ey[i_y == 0] = np.nan

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
        """
        Generate the real pupil aberration data.

        Returns:
            dict: The pupil aberration data.
        """
        stop_idx = self.optic.surface_group.stop_index

        data = {'Px': np.linspace(-1, 1, self.num_points),
                'Py': np.linspace(-1, 1, self.num_points)}
        for field in self.fields:
            Hx = field[0]
            Hy = field[1]

            data[f'{field}'] = {}
            for wavelength in self.wavelengths:
                data[f'{field}'][f'{wavelength}'] = {}

                self.optic.trace(Hx=Hx, Hy=Hy,
                                 wavelength=wavelength,
                                 num_rays=self.num_points,
                                 distribution='line_x')
                data[f'{field}'][f'{wavelength}']['x'] = \
                    self.optic.surface_group.x[stop_idx, :]
                data[f'{field}'][f'{wavelength}']['intensity_x'] = \
                    self.optic.surface_group.intensity[stop_idx, :]

                self.optic.trace(Hx=Hx, Hy=Hy,
                                 wavelength=wavelength,
                                 num_rays=self.num_points,
                                 distribution='line_y')
                data[f'{field}'][f'{wavelength}']['y'] = \
                    self.optic.surface_group.y[stop_idx, :]
                data[f'{field}'][f'{wavelength}']['intensity_y'] = \
                    self.optic.surface_group.intensity[stop_idx, :]
        return data

    def _generate_paraxial_data(self):
        """
        Generate the paraxial pupil aberration data.

        Returns:
            dict: The pupil aberration data.
        """
        pass
