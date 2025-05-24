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

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.zernike import ZernikeFit


@dataclass
class WavefrontData:
    """
    Data container for wavefront results at a given field and wavelength.

    Attributes:
        pupil_x (be.ndarray): x-coordinates of ray intersections at exit pupil.
        pupil_y (be.ndarray): y-coordinates of ray intersections at exit pupil.
        pupil_z (be.ndarray): z-coordinates of ray intersections at exit pupil.
        opd (be.ndarray): Optical path difference data, normalized to waves.
        intensity (be.ndarray): Ray intensities at the exit pupil.
        radius (be.ndarray): Radius of curvature of the exit pupil reference sphere.
    """

    pupil_x: be.ndarray
    pupil_y: be.ndarray
    pupil_z: be.ndarray
    opd: be.ndarray
    intensity: be.ndarray
    radius: be.ndarray


class Wavefront:
    """
    Performs wavefront analysis on an optical system.

    Computes ray intersection points with the exit pupil, the optical path
    difference (OPD) relative to a reference sphere, the radius of curvature
    of the exit-pupil reference sphere, and ray intensities.

    Args:
        optic (Optic): The optical system to analyze.
        fields (str or List[Tuple[float, float]]): Fields or 'all'.
        wavelengths (str or List[float]): Wavelengths or 'all'/'primary'.
        num_rays (int): Number of rays for pupil sampling.
        distribution (str or Distribution): Ray distribution or its name.

    Attributes:
        data (List[List[WavefrontData]]): Nested lists indexed
            by [field][wavelength].
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
        self.fields = self._resolve_fields(fields)
        self.wavelengths = self._resolve_wavelengths(wavelengths)
        self.num_rays = num_rays
        self.distribution = self._resolve_distribution(distribution, self.num_rays)
        self.data = {}
        self._generate_data()

    def get_data(self, field, wl):
        """
        Retrieve precomputed wavefront data for a given field and wavelength.

        Args:
            field (Tuple[float, float]): Field coordinates.
            wl (float): Wavelength.

        Returns:
            WavefrontData: Data container with intersections, OPD, intensity, and
            curvature.
        """
        return self.data[(field, wl)]

    def _resolve_fields(self, fields):
        """Resolve field coordinates based on input."""
        if fields == "all":
            return self.optic.fields.get_field_coords()
        return fields

    def _resolve_wavelengths(self, wavelengths):
        """Resolve wavelengths based on input."""
        if wavelengths == "all":
            return self.optic.wavelengths.get_wavelengths()
        if wavelengths == "primary":
            return [self.optic.primary_wavelength]
        return wavelengths

    def _resolve_distribution(self, dist, num_rays):
        """Resolve distribution based on input."""
        if isinstance(dist, str):
            dist_obj = create_distribution(dist)
            dist_obj.generate_points(num_rays)
            return dist_obj
        return dist

    def _generate_data(self):
        """Generate wavefront data for all fields and wavelengths."""
        pupil_z = self.optic.paraxial.XPL() + self.optic.surface_group.positions[-1]
        for field in self.fields:
            for wl in self.wavelengths:
                # trace chief ray and get reference sphere
                self._trace_chief_ray(field, wl)
                xc, yc, zc, R = self._get_reference_sphere(pupil_z)

                # reference OPD (chief ray)
                opd_ref, _ = self._get_path_length(xc, yc, zc, R, wl)
                opd_ref = self._correct_tilt(field, opd_ref, x=0, y=0)

                # generate full field data
                self.data[(field, wl)] = self._generate_field_data(
                    field, wl, opd_ref, xc, yc, zc, R
                )

    def _generate_field_data(self, field, wavelength, opd_ref, xc, yc, zc, R):
        """
        Generate WavefrontData for a single field and wavelength.

        Args:
            field (tuple): Field coordinates.
            wavelength (float): Wavelength.
            opd_ref: Reference OPD from chief ray.
            xc, yc, zc: Reference sphere center.
            R: Reference sphere radius.

        Returns:
            WavefrontData: All per-ray results.
        """
        rays = self.optic.trace(*field, wavelength, None, self.distribution)
        intensity = self.optic.surface_group.intensity[-1, :]

        opd, t = self._get_path_length(xc, yc, zc, R, wavelength)
        opd = self._correct_tilt(field, opd)
        opd_wv = (opd_ref - opd) / (wavelength * 1e-3)  # OPD map in waves

        pupil_x = rays.x - t * rays.L
        pupil_y = rays.y - t * rays.M
        pupil_z = rays.z - t * rays.N
        return WavefrontData(
            pupil_x=pupil_x,
            pupil_y=pupil_y,
            pupil_z=pupil_z,
            opd=opd_wv,
            intensity=intensity,
            radius=R,
        )

    def _trace_chief_ray(self, field, wavelength):
        """
        Trace the chief ray for a given field and wavelength.
        """
        self.optic.trace_generic(*field, Px=0.0, Py=0.0, wavelength=wavelength)

    def _get_reference_sphere(self, pupil_z):
        """
        Determine reference sphere center and radius from chief ray.
        """
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        z = self.optic.surface_group.z[-1, :]
        if be.size(x) != 1:
            raise ValueError("Chief ray cannot be determined. It must be traced alone.")
        R = be.sqrt(x**2 + y**2 + (z - pupil_z) ** 2)
        return x, y, z, R

    def _get_path_length(self, xc, yc, zc, R, wavelength):
        """
        Calculate optical path difference from image to reference sphere.
        """
        opd_chief = self.optic.surface_group.opd[-1, :]
        opd_img = self._opd_image_to_xp(xc, yc, zc, R, wavelength)
        return opd_chief - opd_img, opd_img

    def _correct_tilt(self, field, opd, x=None, y=None):
        """
        Correct tilt in OPD based on field angle and distribution.
        """
        correction = 0
        if self.optic.field_type == "angle":
            Hx, Hy = field
            max_f = self.optic.fields.max_field
            x_tilt = max_f * Hx
            y_tilt = max_f * Hy
            xs = self.distribution.x if x is None else x
            ys = self.distribution.y if y is None else y
            EPD = self.optic.paraxial.EPD()
            correction = (1 - xs) * be.sin(be.radians(x_tilt)) * EPD / 2 + (
                1 - ys
            ) * be.sin(be.radians(y_tilt)) * EPD / 2
        return opd - correction

    def _opd_image_to_xp(self, xc, yc, zc, R, wavelength):
        """
        Compute propagation distance from image plane to exit pupil.
        """
        xr = self.optic.surface_group.x[-1, :]
        yr = self.optic.surface_group.y[-1, :]
        zr = self.optic.surface_group.z[-1, :]
        L = -self.optic.surface_group.L[-1, :]
        M = -self.optic.surface_group.M[-1, :]
        N = -self.optic.surface_group.N[-1, :]
        a = L**2 + M**2 + N**2
        b = 2 * (L * (xr - xc) + M * (yr - yc) + N * (zr - zc))
        c = (
            xr**2
            + yr**2
            + zr**2
            - 2 * (xr * xc + yr * yc + zr * zc)
            + xc**2
            + yc**2
            + zc**2
            - R**2
        )
        d = b**2 - 4 * a * c
        t = (-b - be.sqrt(d)) / (2 * a)
        mask = t < 0
        t[mask] = (-b[mask] + be.sqrt(d[mask])) / (2 * a[mask])
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
        axs = np.atleast_2d(axs)

        for i, field in enumerate(self.fields):
            for wavelength in self.wavelengths:
                data = self.get_data(field, wavelength)

                wx = data.opd[self.num_rays :]
                wy = data.opd[: self.num_rays]

                intensity_x = data.intensity[self.num_rays :]
                intensity_y = data.intensity[: self.num_rays]

                wx[intensity_x == 0] = np.nan
                wy[intensity_y == 0] = np.nan

                axs[i, 0].plot(
                    be.to_numpy(self.pupil_coord),
                    be.to_numpy(wy),
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
                    be.to_numpy(self.pupil_coord),
                    be.to_numpy(wy),
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
        data = self.get_data(self.fields[0], self.wavelengths[0])
        return be.sqrt(be.mean(data.opd**2))

    def _plot_2d(self, data, figsize=(7, 5.5)):
        """Plots the 2D visualization of the OPD wavefront.

        Args:
            data (dict): The OPD map data.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).

        """
        _, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(np.flipud(data["z"]), extent=[-1, 1, -1, 1])

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
        data = self.get_data(self.fields[0], self.wavelengths[0])
        x = be.to_numpy(self.distribution.x)
        y = be.to_numpy(self.distribution.y)
        z = be.to_numpy(data.opd)
        intensity = be.to_numpy(data.intensity)

        x_interp, y_interp = np.meshgrid(
            np.linspace(-1, 1, num_points),
            np.linspace(-1, 1, num_points),
        )

        points = np.column_stack((x.flatten(), y.flatten()))
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

        data = self.get_data(self.fields[0], self.wavelengths[0])
        z = data.opd

        ZernikeFit.__init__(self, x, y, z, zernike_type, num_terms)
