"""Wavefront Module

This module defines the `Wavefront` class, which is designed to analyze the
wavefront of an optical system. It supports the evaluation of wavefront
aberrations using Zernike polynomials, the generation of wavefront maps for
different field positions and wavelengths, and the calculation of optical path
differences (OPD) relative to a reference sphere. The module integrates with
the rest of the Optiland suite, utilizing its distribution, Zernike, and
optical system modeling capabilities.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.zernike import ZernikeFit


class Wavefront:
    """Represents a wavefront analysis for an optic.

    Args:
        optic (Optic): The optic on which to perform the wavefront analysis.
        fields (str or list, optional): The fields to analyze.
            Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_rays (int, optional): The number of rays to use for the analysis.
            Defaults to 12.
        distribution (str or Distribution, optional): The distribution of rays.
            Defaults to 'hexapolar'.

    Attributes:
        optic (Optic): The optic object being analyzed.
        fields (list): The fields to analyze, as a list of (x, y) tuples.
        wavelengths (list): The wavelengths to analyze.
        num_rays (int): The number of rays used for the analysis.
        distribution (Distribution): The distribution of rays.
        data (list): The generated wavefront data.

    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelengths="all",
        num_rays=12,
        distribution="hexapolar",
    ):
        self.optic = optic
        self.fields = fields
        self.wavelengths = wavelengths
        self.num_rays = num_rays

        if self.fields == "all":
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelengths == "all":
            self.wavelengths = self.optic.wavelengths.get_wavelengths()
        elif self.wavelengths == "primary":
            self.wavelengths = [optic.primary_wavelength]

        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        self.distribution = distribution

        self.data = self._generate_data(self.fields, self.wavelengths)

    def _generate_data(self, fields, wavelengths):
        """Generates the wavefront data for the specified fields and wavelengths.

        Args:
            fields (list): The fields to analyze.
            wavelengths (list): The wavelengths to analyze.

        Returns:
            list: The generated wavefront data.

        """
        pupil_z = self.optic.paraxial.XPL() + self.optic.surface_group.positions[-1]

        data = []
        for field in fields:
            field_data = []
            for wavelength in wavelengths:
                # Trace chief ray for field & find reference sphere properties
                self._trace_chief_ray(field, wavelength)

                # Reference sphere center and radius
                xc, yc, zc, R = self._get_reference_sphere(pupil_z)
                opd_ref = self._get_path_length(xc, yc, zc, R, wavelength)
                opd_ref = self._correct_tilt(field, opd_ref, x=0, y=0)

                field_data.append(
                    self._generate_field_data(
                        field, wavelength, opd_ref, xc, yc, zc, R
                    ),
                )
            data.append(field_data)
        return data

    def _generate_field_data(self, field, wavelength, opd_ref, xc, yc, zc, R):
        """Generates the wavefront data for a specific field and wavelength.

        Args:
            field (tuple): The field coordinates.
            wavelength (float): The wavelength.
            opd_ref (float): The reference optical path length.
            xc (float): The x-coordinate of the reference sphere center.
            yc (float): The y-coordinate of the reference sphere center.
            zc (float): The z-coordinate of the reference sphere center.
            R (float): The radius of the reference sphere.

        Returns:
            tuple: The generated wavefront data, including the optical path
                difference and intensity.

        """
        # trace distribution through pupil
        self.optic.trace(*field, wavelength, None, self.distribution)
        intensity = self.optic.surface_group.intensity[-1, :]
        opd = self._get_path_length(xc, yc, zc, R, wavelength)
        opd = self._correct_tilt(field, opd)
        return (opd_ref - opd) / (wavelength * 1e-3), intensity

    def _trace_chief_ray(self, field, wavelength):
        """Traces the chief ray for a specific field and wavelength.

        Args:
            field (tuple): The field coordinates.
            wavelength (float): The wavelength.

        """
        self.optic.trace_generic(*field, Px=0.0, Py=0.0, wavelength=wavelength)

    def _get_reference_sphere(self, pupil_z):
        """Calculates the properties of the reference sphere.

        Args:
            pupil_z (float): The z-coordinate of the pupil.

        Returns:
            tuple: The x-coordinate, y-coordinate, z-coordinate, and radius of
                the reference sphere.

        Raises:
            ValueError: If the chief ray cannot be determined.

        """
        if self.optic.surface_group.x[-1, :].size != 1:
            raise ValueError("Chief ray cannot be determined. It must be traced alone.")

        # chief ray intersection location
        xc = self.optic.surface_group.x[-1, :]
        yc = self.optic.surface_group.y[-1, :]
        zc = self.optic.surface_group.z[-1, :]

        # radius of sphere - exit pupil origin vs. center
        R = be.sqrt(xc**2 + yc**2 + (zc - pupil_z) ** 2)

        return xc, yc, zc, R

    def _get_path_length(self, xc, yc, zc, r, wavelength):
        """Calculates the optical path difference.

        Args:
            xc (float): The x-coordinate of the reference sphere center.
            yc (float): The y-coordinate of the reference sphere center.
            zc (float): The z-coordinate of the reference sphere center.
            r (float): The radius of the reference sphere.
            wavelength (float): The wavelength of the light.

        Returns:
            float: The optical path difference.

        """
        opd = self.optic.surface_group.opd[-1, :]
        return opd - self._opd_image_to_xp(xc, yc, zc, r, wavelength)

    def _correct_tilt(self, field, opd, x=None, y=None):
        """Corrects for tilt in the optical path difference.

        Args:
            field (tuple): The field coordinates.
            opd (float): The optical path difference.
            x (float, optional): The x-coordinate. Defaults to None.
            y (float, optional): The y-coordinate. Defaults to None.

        Returns:
            float: The corrected optical path difference.

        """
        tilt_correction = 0
        if self.optic.field_type == "angle":
            Hx, Hy = field
            max_field = self.optic.fields.max_field
            x_tilt = max_field * Hx
            y_tilt = max_field * Hy
            if x is None:
                x = self.distribution.x
            if y is None:
                y = self.distribution.y
            EPD = self.optic.paraxial.EPD()
            tilt_correction = (1 - x) * be.sin(be.radians(x_tilt)) * EPD / 2 + (
                1 - y
            ) * be.sin(be.radians(y_tilt)) * EPD / 2
        return opd - tilt_correction

    def _opd_image_to_xp(self, xc, yc, zc, R, wavelength):
        """Finds propagation distance from image plane to reference sphere.

        Args:
            xc (float): The x-coordinate of the reference sphere center.
            yc (float): The y-coordinate of the reference sphere center.
            zc (float): The z-coordinate of the reference sphere center.
            R (float): The radius of the reference sphere.
            wavelength (float): The wavelength of the light.

        Returns:
            float: Propagation distance from image plane to reference sphere.

        """
        xr = self.optic.surface_group.x[-1, :]
        yr = self.optic.surface_group.y[-1, :]
        zr = self.optic.surface_group.z[-1, :]

        L = -self.optic.surface_group.L[-1, :]
        M = -self.optic.surface_group.M[-1, :]
        N = -self.optic.surface_group.N[-1, :]

        a = L**2 + M**2 + N**2
        b = 2 * L * (xr - xc) + 2 * M * (yr - yc) + 2 * N * (zr - zc)
        c = (
            xr**2
            + yr**2
            + zr**2
            - 2 * xr * xc
            + xc**2
            - 2 * yr * yc
            + yc**2
            - 2 * zr * zc
            + zc**2
            - R**2
        )

        d = b**2 - 4 * a * c
        t = (-b - be.sqrt(d)) / (2 * a)
        t[t < 0] = (-b[t < 0] + be.sqrt(d[t < 0])) / (2 * a[t < 0])

        # refractive index in image space
        n = self.optic.image_surface.material_post.n(wavelength)
        return n * t


class OPDFan(Wavefront):
    """Represents a fan plot of the wavefront error for a given optic.

    Args:
        optic (Optic): The optic for which the wavefront error is calculated.
        fields (str or list, optional): The fields for which the wavefront
            error is calculated. Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths for which the
            wavefront error is calculated. Defaults to 'all'.
        num_rays (int, optional): The number of rays used to calculate the
            wavefront error. Defaults to 100.

    Attributes:
        pupil_coord (numpy.ndarray): The coordinates of the pupil.
        data (numpy.ndarray): The wavefront error data.

    Methods:
        view: Plots the wavefront error.

    """

    def __init__(self, optic, fields="all", wavelengths="all", num_rays=100):
        self.pupil_coord = be.linspace(-1, 1, num_rays)
        super().__init__(
            optic,
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution="cross",
        )

    def view(self, figsize=(10, 3)):
        """Visualizes the wavefront error for different fields and wavelengths.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (10, 3).

        """
        num_rows = len(self.fields)

        _, axs = plt.subplots(
            nrows=len(self.fields),
            ncols=2,
            figsize=(figsize[0], num_rows * figsize[1]),
            sharex=True,
            sharey=True,
        )

        # assure axes is a 2D array
        axs = be.atleast_2d(axs)

        for i, field in enumerate(self.fields):
            for j, wavelength in enumerate(self.wavelengths):
                wx = self.data[i][j][0][self.num_rays :]
                wy = self.data[i][j][0][: self.num_rays]

                intensity_x = self.data[i][j][1][self.num_rays :]
                intensity_y = self.data[i][j][1][: self.num_rays]

                wx[intensity_x == 0] = be.nan
                wy[intensity_y == 0] = be.nan

                axs[i, 0].plot(
                    self.pupil_coord,
                    wy,
                    zorder=3,
                    label=f"{wavelength:.4f} µm",
                )
                axs[i, 0].grid()
                axs[i, 0].axhline(y=0, lw=1, color="gray")
                axs[i, 0].axvline(x=0, lw=1, color="gray")
                axs[i, 0].set_xlabel("$P_y$")
                axs[i, 0].set_ylabel("Wavefront Error (waves)")
                axs[i, 0].set_xlim((-1, 1))
                axs[i, 0].set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

                axs[i, 1].plot(
                    self.pupil_coord,
                    wx,
                    zorder=3,
                    label=f"{wavelength:.4f} µm",
                )
                axs[i, 1].grid()
                axs[i, 1].axhline(y=0, lw=1, color="gray")
                axs[i, 1].axvline(x=0, lw=1, color="gray")
                axs[i, 1].set_xlabel("$P_x$")
                axs[i, 1].set_ylabel("Wavefront Error (waves)")
                axs[i, 0].set_xlim((-1, 1))
                axs[i, 1].set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.subplots_adjust(top=1)
        plt.tight_layout()
        plt.show()


class OPD(Wavefront):
    """Represents an Optical Path Difference (OPD) wavefront.

    Args:
        optic (Optic): The optic object.
        field (tuple): The field at which to calculate the OPD.
        wavelength (float): The wavelength of the wavefront.
        num_rings (int, optional): The number of rings for ray tracing.
            Defaults to 15.

    Attributes:
        optic (Optic): The optic object.
        field (Field): The field object.
        wavelength (float): The wavelength of the wavefront.
        num_rings (int): The number of rings for ray tracing.
        distribution (str): The distribution type for ray tracing.
        data (ndarray): The wavefront data.

    Methods:
        view(projection='2d', num_points=256, figsize=(7, 5.5)): Visualizes
            the OPD wavefront.
        rms(): Calculates the root mean square (RMS) of the OPD wavefront.

    """

    def __init__(self, optic, field, wavelength, num_rings=15):
        super().__init__(
            optic,
            fields=[field],
            wavelengths=[wavelength],
            num_rays=num_rings,
            distribution="hexapolar",
        )

    def view(self, projection="2d", num_points=256, figsize=(7, 5.5)):
        """Visualizes the OPD wavefront.

        Args:
            projection (str, optional): The projection type. Defaults to '2d'.
            num_points (int, optional): The number of points for interpolation.
                Defaults to 256.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).

        Raises:
            ValueError: If the projection is not '2d' or '3d'.

        """
        opd_map = self.generate_opd_map(num_points)
        if projection == "2d":
            self._plot_2d(data=opd_map, figsize=figsize)
        elif projection == "3d":
            self._plot_3d(data=opd_map, figsize=figsize)
        else:
            raise ValueError('OPD projection must be "2d" or "3d".')

    def rms(self):
        """Calculates the root mean square (RMS) of the OPD wavefront.

        Returns:
            float: The RMS value.

        """
        return be.sqrt(be.mean(self.data[0][0][0] ** 2))

    def _plot_2d(self, data, figsize=(7, 5.5)):
        """Plots the 2D visualization of the OPD wavefront.

        Args:
            data (dict): The OPD map data.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).

        """
        _, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(be.flipud(data["z"]), extent=[-1, 1, -1, 1])

        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_title(f"OPD Map: RMS={self.rms():.3f} waves")

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("OPD (waves)", rotation=270)
        plt.show()

    def _plot_3d(self, data, figsize=(7, 5.5)):
        """Plots the 3D visualization of the OPD wavefront.

        Args:
            data (dict): The OPD map data.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).

        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)

        surf = ax.plot_surface(
            data["x"],
            data["y"],
            data["z"],
            rstride=1,
            cstride=1,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
        )

        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_zlabel("OPD (waves)")
        ax.set_title(f"OPD Map: RMS={self.rms():.3f} waves")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15)
        fig.tight_layout()
        plt.show()

    def generate_opd_map(self, num_points=256):
        """Generates the OPD map data.

        Args:
            num_points (int, optional): The number of points for interpolation.
                Defaults to 256.

        Returns:
            dict: The OPD map data.

        """
        x = self.distribution.x
        y = self.distribution.y
        z = self.data[0][0][0]
        intensity = self.data[0][0][1]

        x_interp, y_interp = be.meshgrid(
            be.linspace(-1, 1, num_points),
            be.linspace(-1, 1, num_points),
        )

        points = be.column_stack((x.flatten(), y.flatten()))
        values = z.flatten() * intensity.flatten()

        z_interp = griddata(points, values, (x_interp, y_interp), method="cubic")

        data = dict(x=x_interp, y=y_interp, z=z_interp)
        return data


class ZernikeOPD(ZernikeFit, OPD):
    """Represents a Zernike Optical Path Difference (OPD) calculation.

    This class inherits from both the ZernikeFit and OPD classes. It first
    generates the OPD map(s), then fits Zernike polynomials to the map(s).

    Args:
        optic (object): The optic object representing the optical system.
        field (tuple): The field used for the calculation.
        wavelength (float): The wavelength of light used in the calculation.
        num_rings (int, optional): The number of rings used in the Zernike
            calculation. Default is 15.
        zernike_type (str, optional): The type of Zernike polynomials used.
            Default is 'fringe'. See zernike module for more information.
        num_terms (int, optional): The number of Zernike terms used in the
            calculation. Default is 37.

    """

    def __init__(
        self,
        optic,
        field,
        wavelength,
        num_rings=15,
        zernike_type="fringe",
        num_terms=37,
    ):
        OPD.__init__(self, optic, field, wavelength, num_rings)

        x = self.distribution.x
        y = self.distribution.y
        z = self.data[0][0][0]

        ZernikeFit.__init__(self, x, y, z, zernike_type, num_terms)
