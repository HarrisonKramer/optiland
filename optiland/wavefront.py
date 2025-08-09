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
from optiland.fields.field_modes import AngleFieldMode
from optiland.zernike import ZernikeFit


@dataclass
class WavefrontData:
    """Data container for wavefront results at a given field and wavelength.

    Attributes:
        pupil_x (be.ndarray): x-coordinates of ray intersections at exit pupil.
        pupil_y (be.ndarray): y-coordinates of ray intersections at exit pupil.
        pupil_z (be.ndarray): z-coordinates of ray intersections at exit pupil.
        opd (be.ndarray): Optical path difference data, normalized to waves.
        intensity (be.ndarray): Ray intensities at the exit pupil.
        radius (be.ndarray): Radius of curvature of the exit pupil reference sphere.

    """

    pupil_x: be.ndarray  # type: ignore
    pupil_y: be.ndarray  # type: ignore
    pupil_z: be.ndarray  # type: ignore
    opd: be.ndarray  # type: ignore
    intensity: be.ndarray  # type: ignore
    radius: be.ndarray  # type: ignore


class Wavefront:
    """Performs wavefront analysis on an optical system.

    Computes ray intersection points with the exit pupil, the optical path
    difference (OPD) relative to a reference sphere, the radius of curvature
    of the exit-pupil reference sphere, and ray intensities.

    Args:
        optic (Optic): The optical system to analyze.
        fields (str | list[tuple[float, float]]): Fields to analyze.
            Can be 'all' for all defined fields in the optic, or a list of
            (Hx, Hy) field coordinate tuples.
        wavelengths (str | list[float]): Wavelengths to analyze.
            Can be 'all' for all defined wavelengths, 'primary' for the
            primary wavelength, or a list of wavelength values in microns.
        num_rays (int): Number of rays for pupil sampling.
        distribution (str | optiland.distribution.BaseDistribution):
            Ray distribution pattern for pupil sampling. Can be a string name
            (e.g., "hexapolar", "grid") or a BaseDistribution instance.

    Attributes:
        optic (Optic): The optical system being analyzed.
        fields (list[tuple[float, float]]): Resolved list of field coordinates.
        wavelengths (list[float]): Resolved list of wavelengths.
        num_rays (int): Number of rays used for sampling.
        distribution (optiland.distribution.BaseDistribution): Distribution instance.
        data (dict[tuple[tuple[float, float], float], WavefrontData]):
            A dictionary mapping (field_tuple, wavelength) keys to
            WavefrontData objects containing the analysis results.

    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelengths="all",
        num_rays=12,
        distribution="hexapolar",
    ):
        """Initialize Wavefront analysis.

        Args:
            optic (Optic): The optical system to analyze.
            fields (str | list[tuple[float, float]], optional): Fields to analyze.
                Defaults to 'all'.
            wavelengths (str | list[float], optional): Wavelengths to analyze.
                Defaults to 'all'.
            num_rays (int, optional): Number of rays for pupil sampling.
                Defaults to 12.
            distribution (str | optiland.distribution.BaseDistribution, optional):
                Ray distribution. Defaults to "hexapolar".

        """
        self.optic = optic
        self.fields = self._resolve_fields(fields)
        self.wavelengths = self._resolve_wavelengths(wavelengths)
        self.num_rays = num_rays
        self.distribution = self._resolve_distribution(distribution, self.num_rays)
        self.data = {}
        self._generate_data()

    def get_data(self, field, wl):
        """Retrieve precomputed wavefront data for a given field and wavelength.

        Args:
            field (tuple[float, float]): Field coordinates (Hx, Hy).
            wl (float): Wavelength in microns.

        Returns:
            WavefrontData: Data container with intersections, OPD, intensity,
            and curvature for the specified field and wavelength.

        """
        return self.data[(field, wl)]

    def _resolve_fields(self, fields_in):
        """Resolve field coordinates based on input."""
        if fields_in == "all":
            return self.optic.fields.get_field_coords()
        return fields_in

    def _resolve_wavelengths(self, wavelengths_in):
        """Resolve wavelengths based on input."""
        if wavelengths_in == "all":
            return self.optic.wavelengths.get_wavelengths()
        if wavelengths_in == "primary":
            return [self.optic.primary_wavelength]
        return wavelengths_in

    def _resolve_distribution(self, dist_in, num_rays):
        """Resolve distribution based on input."""
        if isinstance(dist_in, str):
            dist_obj = create_distribution(dist_in)
            dist_obj.generate_points(num_rays)
            return dist_obj
        return dist_in

    def _generate_data(self):
        """Generate wavefront data for all specified fields and wavelengths."""
        # Global z-coordinate of the exit pupil
        pupil_z = self.optic.paraxial.XPL() + self.optic.surface_group.positions[-1]
        for field_coord in self.fields:
            for wl_val in self.wavelengths:
                # Trace chief ray to establish reference sphere parameters
                self._trace_chief_ray(field_coord, wl_val)
                xc, yc, zc, R = self._get_reference_sphere(pupil_z)

                # Reference OPD (from chief ray) and tilt correction
                opd_ref, _ = self._get_path_length(xc, yc, zc, R, wl_val)
                opd_ref = self._correct_tilt(field_coord, opd_ref, x=0, y=0)

                # Generate and store full field wavefront data
                self.data[(field_coord, wl_val)] = self._generate_field_data(
                    field_coord, wl_val, opd_ref, xc, yc, zc, R
                )

    def _generate_field_data(self, field, wavelength, opd_ref, xc, yc, zc, R):
        """Generate WavefrontData for a single field and wavelength.

        Args:
            field (tuple[float, float]): Field coordinates (Hx, Hy).
            wavelength (float): Wavelength in microns.
            opd_ref (float): Reference OPD from the chief ray.
            xc (float): X-coordinate of the reference sphere center.
            yc (float): Y-coordinate of the reference sphere center.
            zc (float): Z-coordinate of the reference sphere center.
            R (float): Radius of the reference sphere.

        Returns:
            WavefrontData: Object containing all per-ray results (pupil
            coordinates, OPD, intensity, reference sphere radius).

        """
        rays = self.optic.trace(*field, wavelength, None, self.distribution)
        intensity = self.optic.surface_group.intensity[-1, :]

        opd_values, t_values = self._get_path_length(xc, yc, zc, R, wavelength)
        opd_corrected = self._correct_tilt(field, opd_values)
        # OPD map in waves (microns to waves conversion)
        opd_wv = (opd_ref - opd_corrected) / (wavelength * 1e-3)

        pupil_x = rays.x - t_values * rays.L
        pupil_y = rays.y - t_values * rays.M
        pupil_z = rays.z - t_values * rays.N
        return WavefrontData(
            pupil_x=pupil_x,
            pupil_y=pupil_y,
            pupil_z=pupil_z,
            opd=opd_wv,
            intensity=intensity,
            radius=R,
        )

    def _trace_chief_ray(self, field, wavelength):
        """Trace the chief ray for a given field and wavelength."""
        self.optic.trace_generic(*field, Px=0.0, Py=0.0, wavelength=wavelength)

    def _get_reference_sphere(self, pupil_z):
        """Determine reference sphere center and radius from chief ray."""
        # Assumes chief ray (single ray) was last traced
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        z = self.optic.surface_group.z[-1, :]
        if be.size(x) != 1:  # Should be a single point for chief ray
            raise ValueError(
                "Reference sphere calculation expects a single chief ray to "
                "have been traced."
            )
        # Radius of reference sphere centered at (x,y,z) passing through pupil_z on axis
        R = be.sqrt(x**2 + y**2 + (z - pupil_z) ** 2)
        return x, y, z, R  # Sphere center is (x,y,z) of chief ray on image

    def _get_path_length(self, xc, yc, zc, R, wavelength):
        """Calculate optical path difference from image to reference sphere."""
        opd_chief = self.optic.surface_group.opd[-1, :]  # OPD to image plane
        # opd_img_to_xp is path length from image plane to reference sphere
        opd_img_to_xp, t_values = self._opd_image_to_xp(xc, yc, zc, R, wavelength)
        # Total OPD to ref sphere = OPD to image - path from image to ref sphere
        return opd_chief - opd_img_to_xp, t_values

    def _correct_tilt(self, field, opd, x=None, y=None):
        """Correct tilt in OPD based on field angle and distribution."""
        correction = 0
        if isinstance(self.optic.fields.mode, AngleFieldMode):
            Hx, Hy = field
            max_f = self.optic.fields.max_field  # Max field angle in degrees
            x_tilt_angle = max_f * Hx
            y_tilt_angle = max_f * Hy

            xs = self.distribution.x if x is None else x
            ys = self.distribution.y if y is None else y

            EPD = self.optic.paraxial.EPD()
            correction = (1 - xs) * be.sin(be.radians(x_tilt_angle)) * EPD / 2 + (
                1 - ys
            ) * be.sin(be.radians(y_tilt_angle)) * EPD / 2
        return opd - correction

    def _opd_image_to_xp(self, xc, yc, zc, R, wavelength):
        """Compute path length from image plane point to reference sphere.

        This calculates the distance 't' along the ray from its intersection
        with the image plane to the reference sphere. The path length is n*t.

        Args:
            xc (float): X-coordinate of the reference sphere center.
            yc (float): Y-coordinate of the reference sphere center.
            zc (float): Z-coordinate of the reference sphere center.
            R (float): Radius of the reference sphere.
            wavelength (float): Wavelength for refractive index.

        Returns:
            tuple[be.ndarray, be.ndarray]:
                - Optical path length (n*t) from image to reference sphere.
                - Geometric distance (t) from image to reference sphere.

        """
        # Ray data at the image surface (assuming last traced rays)
        xr = self.optic.surface_group.x[-1, :]
        yr = self.optic.surface_group.y[-1, :]
        zr = self.optic.surface_group.z[-1, :]
        # Direction cosines FROM image plane TOWARDS exit pupil (reverse of ray)
        L = -self.optic.surface_group.L[-1, :]
        M = -self.optic.surface_group.M[-1, :]
        N = -self.optic.surface_group.N[-1, :]

        a_coeff = L**2 + M**2 + N**2
        b_coeff = 2 * (L * (xr - xc) + M * (yr - yc) + N * (zr - zc))
        c_coeff = (xr - xc) ** 2 + (yr - yc) ** 2 + (zr - zc) ** 2 - R**2

        discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
        # Ensure discriminant is non-negative for real solutions
        discriminant = be.maximum(discriminant, be.array(0))

        t1 = (-b_coeff - be.sqrt(discriminant)) / (2 * a_coeff)
        t2 = (-b_coeff + be.sqrt(discriminant)) / (2 * a_coeff)

        t_final = be.where(t1 < 0, t2, t1)  # Prefer t1 if positive, else t2
        t_val = (-b_coeff - be.sqrt(discriminant)) / (2 * a_coeff)
        mask_t_neg = t_val < 0
        t_final = be.where(
            mask_t_neg, (-b_coeff + be.sqrt(discriminant)) / (2 * a_coeff), t_val
        )

        n_val = self.optic.image_surface.material_post.n(wavelength)
        return n_val * t_final, t_final


class OPDFan(Wavefront):
    """Represents a fan plot of the wavefront error for a given optic.

    Args:
        optic (Optic): The optic for which the wavefront error is calculated.
        fields (str | list, optional): The fields for which the wavefront
            error is calculated. Defaults to 'all'.
        wavelengths (str | list, optional): The wavelengths for which the
            wavefront error is calculated. Defaults to 'all'.
        num_rays (int, optional): The number of rays used to calculate the
            wavefront error. Defaults to 100.

    Attributes:
        pupil_coord (be.ndarray): The coordinates of the pupil.

    """

    def __init__(self, optic, fields="all", wavelengths="all", num_rays=100):
        """Initialize OPDFan."""
        self.pupil_coord = be.linspace(-1, 1, num_rays)
        super().__init__(
            optic,
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution="cross",
        )

    def view(
        self, fig_to_plot_on: plt.Figure = None, figsize: tuple[float, float] = (10, 3)
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Visualizes the wavefront error for different fields and wavelengths.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
            figsize (tuple, optional): The size of the figure.
                Defaults to (10, 3).
        Returns:
            tuple: A tuple containing the figure and axes objects.

        Raises:
            ValueError: If the number of fields is not equal to the number of
            wavelengths, or if the number of fields is not equal to the
            number of rays.
        """
        num_rows = len(self.fields)
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            axs = current_fig.add_subplots(
                nrows=len(self.fields),
                ncols=2,
                figsize=(figsize[0], num_rows * figsize[1]),
                sharex=True,
                sharey=True,
            )
        else:
            current_fig, axs = plt.subplots(
                nrows=len(self.fields),
                ncols=2,
                figsize=(figsize[0], num_rows * figsize[1]),
                sharex=True,
                sharey=True,
            )

        axs = np.atleast_2d(axs)  # Ensure axs is 2D for consistent indexing

        for i, field_coord in enumerate(self.fields):
            for wl_val in self.wavelengths:
                data = self.get_data(field_coord, wl_val)

                # Assuming 'cross' distribution: first num_rays are Py, next are Px
                wx = data.opd[self.num_rays :]  # OPD for Px scan (sagittal)
                wy = data.opd[: self.num_rays]  # OPD for Py scan (tangential)

                intensity_x = data.intensity[self.num_rays :]
                intensity_y = data.intensity[: self.num_rays]

                # Set OPD to NaN where intensity is zero (vignetted rays)
                wx = be.where(intensity_x == 0, be.nan, wx)
                wy = be.where(intensity_y == 0, be.nan, wy)

                axs[i, 0].plot(
                    be.to_numpy(self.pupil_coord),
                    be.to_numpy(wy),
                    zorder=3,
                    label=f"{wl_val:.4f} µm",
                )
                axs[i, 0].grid(True)
                axs[i, 0].axhline(y=0, lw=1, color="gray")
                axs[i, 0].axvline(x=0, lw=1, color="gray")
                axs[i, 0].set_xlabel("$P_y$ (Pupil Y-coordinate)")
                axs[i, 0].set_ylabel("Wavefront Error (waves)")
                axs[i, 0].set_xlim((-1, 1))
                axs[i, 0].set_title(
                    f"Hx: {field_coord[0]:.3f}, Hy: {field_coord[1]:.3f} (Tangential)"
                )

                axs[i, 1].plot(
                    be.to_numpy(self.pupil_coord),
                    be.to_numpy(wx),
                    zorder=3,
                    label=f"{wl_val:.4f} µm",
                )
                axs[i, 1].grid(True)
                axs[i, 1].axhline(y=0, lw=1, color="gray")
                axs[i, 1].axvline(x=0, lw=1, color="gray")
                axs[i, 1].set_xlabel("$P_x$")
                axs[i, 1].set_ylabel("Wavefront Error (waves)")
                axs[i, 1].set_xlim((-1, 1))
                axs[i, 1].set_title(
                    f"Hx: {field_coord[0]:.3f}, Hy: {field_coord[1]:.3f}"
                )

        axs[-1, -1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        current_fig.subplots_adjust(top=1)
        current_fig.tight_layout()
        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()

        return current_fig, axs


class OPD(Wavefront):
    """Represents an Optical Path Difference (OPD) wavefront map.

    Args:
        optic (Optic): The optic object.
        field (tuple[float, float]): The field (Hx, Hy) to calculate OPD for.
        wavelength (float): The wavelength of the wavefront in microns.
        num_rings (int, optional): The number of rings for 'hexapolar'
            pupil sampling. Defaults to 15.

    Attributes:
        field (tuple[float, float]): Field coordinates for this OPD map.
        wavelength (float): Wavelength for this OPD map.
        # Other attributes inherited from Wavefront

    """

    def __init__(self, optic, field, wavelength, num_rings=15):
        """Initialize OPD calculation for a specific field and wavelength."""
        super().__init__(
            optic,
            fields=[field],  # OPD is for a single field
            wavelengths=[wavelength],  # and single wavelength
            num_rays=num_rings,  # num_rays for hexapolar is num_rings
            distribution="hexapolar",
        )
        # Store the specific field and wavelength for convenience
        self.field = field
        self.wavelength = wavelength

    def view(
        self,
        fig_to_plot_on: plt.Figure = None,
        projection: str = "2d",
        num_points: int = 256,
        figsize: tuple[float, float] = (7, 5.5),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualizes the OPD wavefront.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure is created.
            projection (str, optional): The projection type. Defaults to '2d'.
            num_points (int, optional): The number of points for interpolation.
                Defaults to 256.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).
        Returns:
            tuple: A tuple containing the figure and axes objects.
        Raises:
            ValueError: If the projection is not '2d' or '3d'.
        """
        is_gui_embedding = fig_to_plot_on is not None
        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = (
                current_fig.add_subplot(111)
                if projection == "2d"
                else current_fig.add_subplot(111, projection="3d")
            )
        else:
            current_fig, ax = (
                plt.subplots(figsize=figsize)
                if projection == "2d"
                else plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
            )

        opd_map = self.generate_opd_map(num_points)
        if projection == "2d":
            self._plot_2d(data=opd_map, ax=ax)
        elif projection == "3d":
            self._plot_3d(fig=current_fig, ax=ax, data=opd_map)
        else:
            raise ValueError('OPD projection must be "2d" or "3d".')

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        return current_fig, ax

    def rms(self):
        """Calculate the root mean square (RMS) of the OPD wavefront.

        Returns:
            float: The RMS wavefront error in waves.

        """
        # Data is stored for self.fields[0] and self.wavelengths[0]
        data = self.get_data(self.field, self.wavelength)
        # Ensure only valid (non-NaN if any) OPD points are used for RMS
        valid_opd = data.opd[~be.isnan(data.opd)]
        if valid_opd.size == 0:
            return 0.0  # Or handle as NaN/error if no valid points
        return be.sqrt(be.mean(valid_opd**2))

    def _plot_2d(self, ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
        """Plots the 2D visualization of the OPD wavefront.

        Args:
            data (dict[str, np.ndarray]): The OPD map data, where keys are 'x', 'y', 'z'
                and values are NumPy arrays suitable for plotting.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).

        """
        im = ax.imshow(
            np.flipud(data["z"]),
            extent=[-1, 1, -1, 1],
            cmap="viridis",
            interpolation="bilinear",
        )

        ax.set_xlabel("Pupil X (normalized)")
        ax.set_ylabel("Pupil Y (normalized)")
        ax.set_title(
            f"OPD Map: Hx={self.field[0]}, Hy={self.field[1]}; "
            f"RMS={self.rms():.3f} waves"
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("OPD (waves)", rotation=270)

    def _plot_3d(
        self, fig: plt.Figure, ax: plt.Axes, data: dict[str, np.ndarray]
    ) -> None:
        """Plots the 3D visualization of the OPD wavefront.

        Args:
            data (dict[str, np.ndarray]): The OPD map data, where keys are 'x', 'y', 'z'
                and values are NumPy arrays suitable for plotting.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).

        """

        # data['x'], data['y'], data['z'] are numpy arrays
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

        ax.set_xlabel("Pupil X (normalized)")
        ax.set_ylabel("Pupil Y (normalized)")
        ax.set_zlabel("OPD (waves)")
        ax.set_title(f"OPD Map: RMS={self.rms():.3f} waves")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15)
        fig.tight_layout()

    def generate_opd_map(self, num_points=256):
        """Generate the OPD map data by interpolating scattered ray data.

        Args:
            num_points (int, optional): The number of points for interpolation
                along each axis of the grid. Defaults to 256.

        Returns:
            dict[str, np.ndarray]: A dictionary containing the interpolated
            OPD map, with keys 'x', 'y', and 'z' (OPD values).
            The values are NumPy arrays.

        """
        # Data is stored for self.fields[0] and self.wavelengths[0]
        wavefront_data_obj = self.get_data(self.field, self.wavelength)

        # Convert backend arrays to NumPy for griddata and plotting
        # Distribution coordinates (normalized pupil)
        x_pupil = be.to_numpy(self.distribution.x).flatten()
        y_pupil = be.to_numpy(self.distribution.y).flatten()
        opd_values = be.to_numpy(wavefront_data_obj.opd).flatten()
        intensity_values = be.to_numpy(wavefront_data_obj.intensity).flatten()

        # Create a mask for valid points (intensity > 0 and OPD is not NaN)
        # Some rays might be vignetted (intensity=0) or fail tracing (opd=NaN)
        valid_mask = (intensity_values > 0) & (~np.isnan(opd_values))

        if not np.any(valid_mask):
            # If no valid points, return an empty or NaN map
            x_interp, y_interp = np.meshgrid(
                np.linspace(-1, 1, num_points),
                np.linspace(-1, 1, num_points),
            )
            return dict(x=x_interp, y=y_interp, z=np.full_like(x_interp, np.nan))

        points_for_griddata = np.column_stack(
            (x_pupil[valid_mask], y_pupil[valid_mask])
        )
        values_for_griddata = opd_values[valid_mask]

        # Create grid for interpolation
        grid_x, grid_y = np.meshgrid(
            np.linspace(-1, 1, num_points),
            np.linspace(-1, 1, num_points),
        )

        # Interpolate OPD data onto the grid
        # Fill un-found points with NaN to represent areas outside pupil or vignetted
        opd_interp = griddata(
            points_for_griddata,
            values_for_griddata,
            (grid_x, grid_y),
            method="cubic",
            fill_value=np.nan,
        )

        # Mask out areas outside the unit circle (pupil boundary)
        pupil_mask = (grid_x**2 + grid_y**2) > 1
        opd_interp[pupil_mask] = np.nan

        return dict(x=grid_x, y=grid_y, z=opd_interp)


class ZernikeOPD(ZernikeFit, OPD):
    """Represents a Zernike polynomial fit to an OPD wavefront.

    This class inherits from both `ZernikeFit` (for fitting capabilities)
    and `OPD` (for OPD calculation). It first calculates the OPD map for a
    given field and wavelength, then fits Zernike polynomials to this OPD data.

    Args:
        optic (Optic): The optical system.
        field (tuple[float, float]): Field coordinates (Hx, Hy).
        wavelength (float): Wavelength in microns.
        num_rings (int, optional): Number of rings for hexapolar pupil
            sampling if generating OPD. Defaults to 15.
        zernike_type (str, optional): Type of Zernike polynomials to use
            (e.g., "fringe", "standard", "noll"). Defaults to "fringe".
        num_terms (int, optional): Number of Zernike terms to fit.
            Defaults to 37.

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
        """Initialize ZernikeOPD analysis."""
        # First, initialize OPD to calculate wavefront data
        OPD.__init__(self, optic, field, wavelength, num_rings)

        # Pupil coordinates from the distribution used by OPD
        # These are normalized pupil coordinates.
        x_pupil = self.distribution.x
        y_pupil = self.distribution.y

        # OPD values from the calculated wavefront data
        # Data is stored for self.fields[0] (which is `field`)
        # and self.wavelengths[0] (which is `wavelength`)
        wavefront_data_obj = self.get_data(field, wavelength)
        opd_values = wavefront_data_obj.opd  # These are in waves

        # Now, initialize ZernikeFit with these pupil coords and OPD values
        ZernikeFit.__init__(self, x_pupil, y_pupil, opd_values, zernike_type, num_terms)
