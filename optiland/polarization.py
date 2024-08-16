import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from optiland.distribution import create_distribution


class JonesPupil:
    """
    Represents a Jones pupil for polarization analysis.

    Args:
        optic (Optic): The optic object.
        field (tuple): The normalized field components (Hx, Hy).
        wavelength (float): The wavelength of the light in microns.
        num_rings (int, optional): The number of rings for generating points
            on a hexpolar grid. Defaults to 15.
        coordinate_basis (str, optional): The coordinate basis. Defaults to
            'double_pole'. Options are 'dipole' or 'double_pole'.

    Raises:
        ValueError: If optic polarization is set to 'ignore'.

    Methods:
        view(num_points=256, figsize=(10, 8)):
            Displays the real and imaginary components of the Jones pupil.

    Attributes:
        x (ndarray): The x-coordinates of the generated points.
        y (ndarray): The y-coordinates of the generated points.
        jones (ndarray): The Jones matrix.
    """

    def __init__(self, optic, field, wavelength, num_rings=15,
                 coordinate_basis='double_pole'):
        self.optic = optic
        self.field = field
        self.wavelength = wavelength
        self.num_rings = num_rings
        self.coordinate_basis = coordinate_basis

        self.x, self.y, self.jones = self._generate_data()

        if optic.polarization == 'ignore':
            raise ValueError('Polarization cannot be set to "ignore" for '
                             'Jones pupil generation.')

    def view(self, num_points=256, figsize=(10, 8)):
        """
        View the interpolated Jones matrix in two 2x2 subplots, representing
        the real and imaginary components of the Jones pupil.

        Args:
            num_points (int): The number of points to interpolate the
                Jones matrix.
            figsize (tuple): The size of the figure in inches.
                Defaults to (10, 8).
        """
        jones = self._interpolate_jones(num_points)
        self.view_real(jones, figsize)
        self.view_imag(jones, figsize)

    def view_real(self, jones, figsize=(10, 8)):
        """
        Plot the real components of the Jones pupil.

        Args:
            jones (dict): A dictionary containing the components of the
                Jones pupil.
            figsize (tuple, optional): The size of the figure.
                Defaults to (10, 8).
        """
        # Plot the real components of the Jones pupil
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes[0, 0].imshow(jones['jxx_real'], extent=(-1, 1, -1, 1))
        axes[0, 0].set_title(r'Jones Pupil: $\Re{(J_{xx})}$')
        axes[0, 1].imshow(jones['jxy_real'], extent=(-1, 1, -1, 1))
        axes[0, 1].set_title(r'Jones Pupil: $\Re{(J_{xy})}$')
        axes[1, 0].imshow(jones['jyx_real'], extent=(-1, 1, -1, 1))
        axes[1, 0].set_title(r'Jones Pupil: $\Re{(J_{yx})}$')
        axes[1, 1].imshow(jones['jyy_real'], extent=(-1, 1, -1, 1))
        axes[1, 1].set_title(r'Jones Pupil: $\Re{(J_{yy})}$')

        for ax in axes.flat:
            ax.set_axis_off()
            ax.margins(0)
            ax.figure.colorbar(ax.images[0], ax=ax)

        plt.tight_layout()
        plt.show()

    def view_imag(self, jones, figsize=(10, 8)):
        """
        Plot the imaginary components of the Jones pupil.
        Args:
            jones (dict): Dictionary containing the components of the
                Jones pupil.
            figsize (tuple, optional): The size of the figure.
                Defaults to (10, 8).
        """
        # Plot the imaginary components of the Jones pupil
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes[0, 0].imshow(jones['jxx_imag'], extent=(-1, 1, -1, 1))
        axes[0, 0].set_title(r'Jones Pupil: $\Im{(J_{xx})}$')
        axes[0, 1].imshow(jones['jxy_imag'], extent=(-1, 1, -1, 1))
        axes[0, 1].set_title(r'Jones Pupil: $\Im{(J_{xy})}$')
        axes[1, 0].imshow(jones['jyx_imag'], extent=(-1, 1, -1, 1))
        axes[1, 0].set_title(r'Jones Pupil: $\Im{(J_{yx})}$')
        axes[1, 1].imshow(jones['jyy_imag'], extent=(-1, 1, -1, 1))
        axes[1, 1].set_title(r'Jones Pupil: $\Im{(J_{yy})}$')

        for ax in axes.flat:
            ax.set_axis_off()
            ax.margins(0)
            ax.figure.colorbar(ax.images[0], ax=ax)

        plt.tight_layout()
        plt.show()

    def _convert_to_local(self, k):
        """Project 3D k-vector into 2D x-y coordinates.

        Args:
            k (numpy.ndarray): 3D k-vector.

        Returns:
            tuple: A tuple containing the x and y coordinates.

        Raises:
            ValueError: If the coordinate basis is unknown.
        """
        kx = k[:, 0]
        ky = k[:, 1]
        kz = k[:, 2]

        if self.coordinate_basis == 'dipole':
            # dipole basis assumes an axis vector of (1, 0, 0)
            x = np.column_stack((np.zeros_like(kx),
                                 kz,
                                 -ky)) / np.sqrt(ky**2 + kz**2)
            y = np.column_stack((ky**2 + kz**2,
                                 -kx * ky,
                                 -kx * kz)) / np.sqrt(ky**2 + kz**2)
        elif self.coordinate_basis == 'double_pole':
            x = np.column_stack((1 - kx**2 / (1 + kz),
                                 -kx * ky / (1 + kz),
                                 -kx))
            y = np.column_stack((-kx * ky / (1 + kz),
                                 1 - ky**2 / (1 + kz),
                                 -ky))
        else:
            raise ValueError(f'Unknown coordinate basis:'
                             f' {self.coordinate_basis}')
        return x, y

    def _generate_data(self):
        """
        Generates data for polarization analysis.

        Returns:
            x (ndarray): pupil x-coordinates of generated points.
            y (ndarray): pupil y-coordinates of generated points.
            jones (ndarray): Jones pupil representing polarization state.
        """
        Hx, Hy = self.field
        vx, vy = self.optic.fields.get_vig_factor(Hx, Hy)

        distribution = create_distribution('hexapolar')
        distribution.generate_points(self.num_rings, vx, vy)
        x = distribution.x
        y = distribution.y
        Px = x * (1 - vx)
        Py = y * (1 - vy)

        rays = self.optic.ray_generator.generate_rays(Hx, Hy, Px, Py,
                                                      self.wavelength)
        k0 = np.column_stack((rays.L, rays.M, rays.N))
        rays = self.optic.surface_group.trace(rays)
        k1 = np.column_stack((rays.L, rays.M, rays.N))

        x0, y0 = self._convert_to_local(k0)
        x1, y1 = self._convert_to_local(k1)

        o_x_inv = np.linalg.inv(np.stack([x0, y0, k0], axis=-1))
        o_e = np.stack([x0, y1, k1], axis=-1)

        jones = np.einsum('ijk,ikl,ilm->ijm', o_x_inv, rays.p, o_e)
        return x, y, jones

    def _interpolate_jones(self, num_points=256):
        x_interp, y_interp = np.meshgrid(np.linspace(-1, 1, num_points),
                                         np.linspace(-1, 1, num_points))
        data = dict(x=x_interp, y=y_interp)

        points = np.column_stack((self.x.flatten(), self.y.flatten()))

        data['jxx_real'] = griddata(points, self.jones[:, 0, 0].real.flatten(),
                                    (x_interp, y_interp), method='cubic')
        data['jxx_imag'] = griddata(points, self.jones[:, 0, 0].imag.flatten(),
                                    (x_interp, y_interp), method='cubic')
        data['jxy_real'] = griddata(points, self.jones[:, 0, 1].real.flatten(),
                                    (x_interp, y_interp), method='cubic')
        data['jxy_imag'] = griddata(points, self.jones[:, 0, 1].imag.flatten(),
                                    (x_interp, y_interp), method='cubic')
        data['jyx_real'] = griddata(points, self.jones[:, 1, 0].real.flatten(),
                                    (x_interp, y_interp), method='cubic')
        data['jyx_imag'] = griddata(points, self.jones[:, 1, 0].imag.flatten(),
                                    (x_interp, y_interp), method='cubic')
        data['jyy_real'] = griddata(points, self.jones[:, 1, 1].real.flatten(),
                                    (x_interp, y_interp), method='cubic')
        data['jyy_imag'] = griddata(points, self.jones[:, 1, 1].imag.flatten(),
                                    (x_interp, y_interp), method='cubic')

        return data
