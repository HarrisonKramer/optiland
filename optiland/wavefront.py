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

    This class computes ray intersection points with the exit pupil, the optical
    path difference (OPD) relative to a reference sphere, the radius of curvature
    of the exit pupil, and ray intensities at the exit pupil.

    Args:
        optic (Optic): Optical system to analyze.
        fields (Union[str, List[Tuple[float, float]]]): Field coordinates or 'all'.
        wavelengths (Union[str, List[float]]): Wavelengths or 'all'/'primary'.
        num_rays (int): Number of rays in the pupil sampling distribution.
        distribution (Union[str, Distribution]): Ray distribution or name.

    Attributes:
        data (Dict[Tuple, WavefrontData]): Nested dict keyed by (field, wavelength).
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
        self._compute_all()

    def _resolve_fields(self, fields):
        if fields == "all":
            return self.optic.fields.get_field_coords()
        return fields

    def _resolve_wavelengths(self, wavelengths):
        if wavelengths == "all":
            return self.optic.wavelengths.get_wavelengths()
        if wavelengths == "primary":
            return [self.optic.primary_wavelength]
        return wavelengths

    def _resolve_distribution(self, dist, num_rays: int):
        if isinstance(dist, str):
            distribution = create_distribution(dist)
            distribution.generate_points(num_rays)
            return distribution
        return dist

    def _compute_all(self) -> None:
        """
        Compute wavefront data for all specified fields and wavelengths.
        """
        # z-coordinate of exit pupil from paraxial data and last surface position
        pupil_z = self.optic.paraxial.XPL() + self.optic.surface_group.positions[-1]

        for field in self.fields:
            for wl in self.wavelengths:
                # trace chief ray and get reference sphere
                self._trace_chief(field, wl)
                xc, yc, zc, R = self._get_reference_sphere(pupil_z)

                # compute propagation distances
                t_chief = self._prop_distance(xc, yc, zc, R)
                rays = self.optic.trace(*field, wl, None, self.distribution)
                t_rays = self._prop_distance(xc, yc, zc, R, rays)

                # compute OPDs and intensities
                opd_ref = self.optic.surface_group.opd[-1, :] - t_chief
                opd = self.optic.surface_group.opd[-1, :] - t_rays
                opd_ref_corr, opd_corr = self._apply_tilt_correction(
                    field, opd_ref, opd
                )

                opd_map = (opd_ref_corr - opd_corr) / (wl * 1e-3)
                intensity = self.optic.surface_group.intensity[-1, :]

                # store results
                self.data[(field, wl)] = WavefrontData(
                    pupil_x=rays.x - t_rays * rays.L,
                    pupil_y=rays.y - t_rays * rays.M,
                    pupil_z=rays.z - t_rays * rays.N,
                    opd=opd_map,
                    intensity=intensity,
                    radius=R,
                )

    def _trace_chief(self, field, wl):
        """
        Trace the chief ray for a given field and wavelength.
        """
        self.optic.trace_generic(*field, Px=0.0, Py=0.0, wavelength=wl)

    def _get_reference_sphere(self, pupil_z: float):
        """
        Determine the reference sphere center and radius from the traced chief ray.
        """
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        z = self.optic.surface_group.z[-1, :]
        if x.size != 1:
            raise ValueError(
                "Chief ray must be traced alone to determine reference sphere."
            )
        R = float(be.sqrt(x**2 + y**2 + (z - pupil_z) ** 2))
        return x, y, z, R

    def _prop_distance(self, xc, yc, zc, R, rays=None):
        """
        Compute propagation distance from image plane or rays to reference sphere.

        If rays is None, uses last surface_group data for chief ray.
        """
        if rays is None:
            xr = self.optic.surface_group.x[-1, :]
            yr = self.optic.surface_group.y[-1, :]
            zr = self.optic.surface_group.z[-1, :]
            L = -self.optic.surface_group.L[-1, :]
            M = -self.optic.surface_group.M[-1, :]
            N = -self.optic.surface_group.N[-1, :]
        else:
            xr, yr, zr = rays.x, rays.y, rays.z
            L, M, N = -rays.L, -rays.M, -rays.N

        # quadratic coefficients
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

        # account for image space refractive index
        n = self.optic.image_surface.material_post.n(
            rays.w if rays else self.optic.primary_wavelength
        )
        return n * t

    def _apply_tilt_correction(self, field, opd_ref, opd):
        """
        Apply tilt correction to OPD arrays based on field angle and sampling
        distribution.
        """
        if self.optic.field_type != "angle":
            return opd_ref, opd

        # angular tilts
        Hx, Hy = field
        max_f = self.optic.fields.max_field
        x_tilt = be.radians(Hx * max_f)
        y_tilt = be.radians(Hy * max_f)
        EPD = self.optic.paraxial.EPD()

        # distribution coordinates
        xs, ys = self.distribution.x, self.distribution.y
        tilt = ((1 - xs) * be.sin(x_tilt) + (1 - ys) * be.sin(y_tilt)) * EPD / 2

        # reference ray is at xs=ys=0, so its tilt is zero by construction
        return opd_ref - tilt, opd - tilt

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
        return be.sqrt(be.mean(self.data[0][0][0] ** 2))

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
        data = self.get_data(self.fields, self.wavelengths[0])
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

        data = self.get_data(self.fields, self.wavelengths[0])
        z = data.opd

        ZernikeFit.__init__(self, x, y, z, zernike_type, num_terms)
