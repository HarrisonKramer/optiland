"""Spot Diagram Analysis

This module provides a spot diagram analysis for optical systems.

Kramer Harrison, 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

import optiland.backend as be
from optiland.utils import resolve_fields
from optiland.visualization.system.utils import transform

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from optiland._types import BEArray, DistributionType


@dataclass
class SpotData:
    """Stores the x, y coordinates and intensity of a spot.

    Attributes:
        x: Array of x-coordinates.
        y: Array of y-coordinates.
        intensity: Array of intensity values.
    """

    x: be.array
    y: be.array
    intensity: be.array


class SpotDiagram(BaseAnalysis):
    """Generates and plots real ray intersection data on the image surface.

    This class creates spot diagrams, which are purely geometric plots that give an
    indication of the blur produced by aberrations in an optical system.

    Attributes:
        optic: Instance of the optic object to be assessed.
        fields: Fields at which data is generated.
        wavelengths: Wavelengths at which data is generated.
        num_rings: Number of rings in the pupil distribution for ray tracing.
        distribution: The pupil distribution type for ray tracing.
        data: Contains spot data in a nested list, ordered by field, then
            wavelength.
        coordinates: The coordinate system ('global' or 'local') for data and
            plotting.
    """

    def __init__(
        self,
        optic,
        fields: str | list = "all",
        wavelengths: str | list = "all",
        num_rings: int = 6,
        distribution: DistributionType = "hexapolar",
        coordinates: Literal["global", "local"] = "local",
    ):
        """Initializes the SpotDiagram analysis.

        Note:
            The constructor generates all data that is later used for plotting.

        Args:
            optic: An instance of the optic object to be assessed.
            fields: Fields at which to generate data. If 'all', all defined
                field points are used. Defaults to "all".
            wavelengths: Wavelengths at which to generate data. If 'all', all
                defined wavelengths are used. Defaults to "all".
            num_rings: Number of rings in the pupil distribution for ray
                tracing. Defaults to 6.
            distribution: Pupil distribution type for ray tracing.
                Defaults to "hexapolar".
            coordinates: Coordinate system for data generation and plotting.
                Defaults to "local".

        Raises:
            ValueError: If `coordinates` is not 'global' or 'local'.
        """
        self.fields = resolve_fields(optic, fields)

        if coordinates not in ["global", "local"]:
            raise ValueError("Coordinates must be 'global' or 'local'.")
        self.coordinates = coordinates

        self.num_rings = num_rings
        self.distribution: DistributionType = distribution

        super().__init__(optic, wavelengths)
        primary_wl_value = self.optic.primary_wavelength
        if primary_wl_value in self.wavelengths:
            # Use the system's primary wavelength as the reference if available.
            self._analysis_ref_wavelength_index = self.wavelengths.index(
                primary_wl_value
            )
        else:
            # Otherwise, use the first wavelength in the analysis list.
            self._analysis_ref_wavelength_index = 0

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (12, 4),
        add_airy_disk: bool = False,
    ) -> tuple[Figure, list[Axes]]:
        """Displays the spot diagram plot.

        Args:
            fig_to_plot_on: An existing Matplotlib figure to plot on. If None,
                a new figure is created. Defaults to None.
            figsize: The figure size for the output window, applied per row.
                Defaults to (12, 4).
            add_airy_disk: If True, adds the Airy disk visualization to the
                plots. Defaults to False.

        Returns:
            A tuple containing the Matplotlib figure and a list of its axes.
        """
        if not self.fields:
            return self._handle_no_fields(fig_to_plot_on)

        # 1. Prepare data for plotting
        centered_data = self._center_spots(self.data)
        airy_disk_data = self._prepare_airy_disk_data() if add_airy_disk else None

        # 2. Set up the figure and calculate plot limits
        fig, axs = self._setup_plot_layout(fig_to_plot_on, figsize)
        axis_lim = self._calculate_axis_limits(centered_data, airy_disk_data)

        # 3. Plot each field on its corresponding subplot
        for i, field_data in enumerate(centered_data):
            if i >= len(axs):
                break
            self._plot_field(
                axs[i], field_data, self.fields[i], axis_lim, i, airy_disk_data
            )

        # 4. Finalize the plot (legend, layout, etc.)
        self._finalize_plot(fig, axs, len(self.fields))

        return fig, fig.get_axes()

    # --- Calculation Methods ---

    def angle_from_cosine(self, a: BEArray, b: BEArray) -> float:
        """Calculates the angle in radians between two direction cosine vectors.

        Args:
            a: The first direction cosine vector.
            b: The second direction cosine vector.

        Returns:
            The angle between the vectors in radians.
        """
        a = a / be.linalg.norm(a)
        b = b / be.linalg.norm(b)
        return be.arccos(be.clip(be.dot(a, b), -1, 1))

    def f_number(self, n: float, theta: float) -> float:
        """Calculates the physical F-number.

        Args:
            n: The refractive index of the medium.
            theta: The half-angle of the cone of light in radians.

        Returns:
            The calculated physical F-number.
        """
        return 1 / (2 * n * be.sin(theta))

    def airy_radius(self, n_w: float, wavelength: float) -> float:
        """Calculates the Airy disk radius.

        Args:
            n_w: The physical F-number.
            wavelength: The wavelength of light in micrometers.

        Returns:
            The Airy disk radius.
        """
        return 1.22 * n_w * wavelength

    def generate_marginal_rays(
        self, H_x: float, H_y: float, wavelength: float
    ) -> tuple:
        """Generates marginal rays at the four cardinal points of the pupil.

        Args:
            H_x: The x-field coordinate.
            H_y: The y-field coordinate.
            wavelength: The wavelength for the rays.

        Returns:
            A tuple containing the traced rays for north, south, east, and west
            pupil points.
        """
        ray_north = self.optic.trace_generic(
            Hx=H_x, Hy=H_y, Px=0, Py=1, wavelength=wavelength
        )
        ray_south = self.optic.trace_generic(
            Hx=H_x, Hy=H_y, Px=0, Py=-1, wavelength=wavelength
        )
        ray_east = self.optic.trace_generic(
            Hx=H_x, Hy=H_y, Px=1, Py=0, wavelength=wavelength
        )
        ray_west = self.optic.trace_generic(
            Hx=H_x, Hy=H_y, Px=-1, Py=0, wavelength=wavelength
        )
        return ray_north, ray_south, ray_east, ray_west

    def generate_marginal_rays_cosines(
        self, H_x: float, H_y: float, wavelength: float
    ) -> tuple:
        """Generates direction cosines for each marginal ray of a given field.

        Args:
            H_x: The x-field coordinate.
            H_y: The y-field coordinate.
            wavelength: The wavelength for the rays.

        Returns:
            A tuple of direction cosine vectors for north, south, east, and west rays.
        """
        rays = self.generate_marginal_rays(H_x, H_y, wavelength)
        return tuple(be.array([ray.L, ray.M, ray.N]).ravel() for ray in rays)

    def generate_chief_rays_cosines(self, wavelength: float) -> BEArray:
        """Generates direction cosines for the chief ray of each field.

        Args:
            wavelength: The wavelength for the rays.

        Returns:
            An array of shape (num_fields, 3) containing the direction cosines.
        """
        cosines = [
            be.array([ray.L, ray.M, ray.N]).ravel()
            for H_x, H_y in self.fields
            for ray in [
                self.optic.trace_generic(
                    Hx=H_x, Hy=H_y, Px=0, Py=0, wavelength=wavelength
                )
            ]
        ]
        return be.stack(cosines, axis=0)

    def generate_chief_rays_centers(self, wavelength: float) -> BEArray:
        """Generates the (x, y) intersection points for the chief ray of each field.

        Args:
            wavelength: The wavelength for the rays.

        Returns:
            An array of shape (num_fields, 2) containing the (x, y) coordinates.
        """
        centers = [
            [ray.x.item(), ray.y.item()]
            for H_x, H_y in self.fields
            for ray in [
                self.optic.trace_generic(
                    Hx=H_x, Hy=H_y, Px=0, Py=0, wavelength=wavelength
                )
            ]
        ]
        return be.stack(centers, axis=0)

    def airy_disc_x_y(self, wavelength: float) -> tuple[list[float], list[float]]:
        """Generates the Airy disk radii for the x and y axes for each field.

        Args:
            wavelength: The wavelength for the calculation.

        Returns:
            A tuple containing two lists: x-axis radii and y-axis radii for each field.
        """
        chief_cosines = self.generate_chief_rays_cosines(wavelength)
        airy_rad_x_list, airy_rad_y_list = [], []

        for i, (H_x, H_y) in enumerate(self.fields):
            north, south, east, west = self.generate_marginal_rays_cosines(
                H_x, H_y, wavelength
            )
            chief = chief_cosines[i]

            angle_x = (
                self.angle_from_cosine(chief, north)
                + self.angle_from_cosine(chief, south)
            ) / 2
            angle_y = (
                self.angle_from_cosine(chief, east)
                + self.angle_from_cosine(chief, west)
            ) / 2

            f_num_x = self.f_number(n=1, theta=angle_x)
            f_num_y = self.f_number(n=1, theta=angle_y)

            # Convert radius from µm to mm
            airy_rad_x_list.append(self.airy_radius(f_num_x, wavelength) * 1e-3)
            airy_rad_y_list.append(self.airy_radius(f_num_y, wavelength) * 1e-3)

        return airy_rad_x_list, airy_rad_y_list

    def centroid(self) -> list[tuple[BEArray, BEArray]]:
        """Calculates the geometric centroid of each spot for the reference wavelength.

        Returns:
            A list of (x, y) centroid coordinates for each field.
        """
        ref_idx = self._analysis_ref_wavelength_index
        return [
            (be.mean(field_data[ref_idx].x), be.mean(field_data[ref_idx].y))
            for field_data in self.data
        ]

    def geometric_spot_radius(self) -> list[list[BEArray]]:
        """Calculates the maximum geometric spot radius for each spot.

        Returns:
            A nested list of maximum radii for each field and wavelength.
        """
        centered_data = self._center_spots(self.data)
        return [
            [
                be.max(be.sqrt(wave_data.x**2 + wave_data.y**2))
                for wave_data in field_data
            ]
            for field_data in centered_data
        ]

    def rms_spot_radius(self) -> list[list[BEArray]]:
        """Calculates the root-mean-square (RMS) spot radius for each spot.

        Returns:
            A nested list of RMS radii for each field and wavelength.
        """
        centered_data = self._center_spots(self.data)
        return [
            [
                be.sqrt(be.mean(wave_data.x**2 + wave_data.y**2))
                for wave_data in field_data
            ]
            for field_data in centered_data
        ]

    # --- Internal Data Generation and Plotting Helpers ---

    def _center_spots(self, data: list[list[SpotData]]) -> list[list[SpotData]]:
        """Centers spot data around the centroid of the reference wavelength.

        Args:
            data: The original, uncentered spot data.

        Returns:
            A deep copy of the data, centered around the respective centroids.
        """
        centroids = self.centroid()
        centered_data = []
        for i, field_list in enumerate(data):
            cx, cy = centroids[i]
            centered_field = [
                SpotData(x=sd.x - cx, y=sd.y - cy, intensity=be.copy(sd.intensity))
                for sd in field_list
            ]
            centered_data.append(centered_field)
        return centered_data

    def _generate_data(self) -> list[list[SpotData]]:
        """Generates spot data for all configured fields and wavelengths.

        Returns:
            A nested list of spot intersection data.
        """
        return [
            [
                self._generate_field_data(
                    field, wl, self.num_rings, self.distribution, self.coordinates
                )
                for wl in self.wavelengths
            ]
            for field in self.fields
        ]

    def _generate_field_data(
        self,
        field: tuple[float, float],
        wavelength: float,
        num_rays: int,
        distribution: DistributionType,
        coordinates: str,
    ) -> SpotData:
        """Generates spot data for a single field and wavelength.

        Args:
            field: The (Hx, Hy) field coordinates.
            wavelength: The wavelength for tracing.
            num_rays: The number of rays to generate, or number of rings if distribution
                is hexapolar.
            distribution: The ray distribution pattern.
            coordinates: The coordinate system ('local' or 'global').

        Returns:
            A SpotData object with the traced ray intersection data.
        """
        self.optic.trace(*field, wavelength, num_rays, distribution)
        surf_group = self.optic.surface_group
        x_g, y_g, z_g, i_g = (
            surf_group.x[-1, :],
            surf_group.y[-1, :],
            surf_group.z[-1, :],
            surf_group.intensity[-1, :],
        )

        # Ignore rays with zero intensity
        mask = i_g > 0
        x_g, y_g, z_g, i_g = x_g[mask], y_g[mask], z_g[mask], i_g[mask]

        if coordinates == "local":
            x_plot, y_plot, _ = transform(
                x_g, y_g, z_g, self.optic.image_surface, is_global=True
            )
        else:
            x_plot, y_plot = x_g, y_g

        return SpotData(x=x_plot, y=y_plot, intensity=i_g)

    def _handle_no_fields(self, fig: Figure) -> tuple[None, None]:
        """Handles the case where there are no fields to plot.

        Args:
            fig: An optional existing figure to draw a message on.
        """
        print("Warning (SpotDiagram.view): No fields to plot.")
        if fig and hasattr(fig, "canvas") and fig.canvas:
            fig.text(
                0.5, 0.5, "No fields to plot Spot Diagram", ha="center", va="center"
            )
            fig.canvas.draw_idle()
        return None, None

    def _setup_plot_layout(
        self, fig_to_plot_on: Figure, figsize: tuple
    ) -> tuple[Figure, NDArray[np.object_]]:
        """Sets up the Matplotlib figure and axes grid.

        Args:
            fig_to_plot_on: An existing figure to use, or None to create one.
            figsize: The size for the figure.

        Returns:
            A tuple of the figure and a flattened array of its axes.
        """
        num_fields = len(self.fields)
        num_cols = 3
        num_rows = (num_fields + num_cols - 1) // num_cols

        if fig_to_plot_on:
            fig = fig_to_plot_on
            fig.clear()
        else:
            fig = plt.figure(figsize=(figsize[0], num_rows * figsize[1]))

        axs = fig.subplots(num_rows, num_cols, sharex=True, sharey=True).flatten()
        return fig, axs

    def _prepare_airy_disk_data(self) -> dict:
        """Prepares all necessary data for plotting the Airy disk.

        Returns:
            A dictionary containing Airy disk radii, chief ray centers, and
            real centroid coordinates.
        """
        primary_wl_obj = self.optic.wavelengths.primary_wavelength
        wl_val = primary_wl_obj.value if primary_wl_obj else self.wavelengths[0]

        airy_rad_x, airy_rad_y = self.airy_disc_x_y(wavelength=wl_val)
        chief_centers = self.generate_chief_rays_centers(wavelength=wl_val)
        geom_centroids = self.centroid()

        real_centroids = be.to_numpy(chief_centers) - be.to_numpy(
            be.stack(geom_centroids)
        )

        return {
            "radii_x": be.to_numpy(airy_rad_x),
            "radii_y": be.to_numpy(airy_rad_y),
            "real_centroids": real_centroids,
        }

    def _calculate_axis_limits(
        self,
        centered_data: list,
        airy_disk_data: dict | None = None,
        buffer: float = 1.05,
    ) -> float:
        """Calculates the axis limits to encompass all spots and Airy disks.

        Args:
            centered_data: The centered spot data.
            airy_disk_data: The prepared Airy disk data, if any.
            buffer: A multiplicative buffer to apply to the final limit.

        Returns:
            A float representing the symmetric axis limit (+/- limit).
        """
        max_radii = [
            be.to_numpy(be.max(be.sqrt(sd.x**2 + sd.y**2)))
            for field in centered_data
            for sd in field
        ]
        max_geom_radius = max(max_radii) if max_radii else 0.01

        if not airy_disk_data:
            return max_geom_radius * buffer

        # If Airy disk is present, consider its size and offset
        rad_x = airy_disk_data["radii_x"]
        rad_y = airy_disk_data["radii_y"]
        centroids = airy_disk_data["real_centroids"]

        max_extent = 0
        for i in range(len(self.fields)):
            offset = np.max(np.abs(centroids[i]))
            radius = max(rad_x[i], rad_y[i], max_geom_radius)
            max_extent = max(max_extent, offset + radius)

        return max_extent * buffer if max_extent > 0 else max_geom_radius * buffer

    def _plot_field(
        self,
        ax: Axes,
        field_data: list[SpotData],
        field_coords: tuple[float, float],
        axis_lim: float,
        field_index: int,
        airy_disk_data: dict | None = None,
    ):
        """Plots the data for a single field on a given axis.

        Args:
            ax: The Matplotlib axis to plot on.
            field_data: A list of SpotData for the current field.
            field_coords: The (Hx, Hy) coordinates of the field.
            axis_lim: The symmetric axis limit for x and y axes.
            field_index: The index of the current field.
            airy_disk_data: Optional dictionary with Airy disk data.
        """
        markers = ["o", "s", "^"]
        for i, points in enumerate(field_data):
            x, y, intensity = (
                be.to_numpy(points.x),
                be.to_numpy(points.y),
                be.to_numpy(points.intensity),
            )
            mask = intensity != 0
            ax.scatter(
                x[mask],
                y[mask],
                s=10,
                label=f"{self.wavelengths[i]:.4f} µm",
                marker=markers[i % 3],
                alpha=0.7,
            )

        if airy_disk_data:
            cx, cy = airy_disk_data["real_centroids"][field_index]
            width = 2 * airy_disk_data["radii_y"][field_index]
            height = 2 * airy_disk_data["radii_x"][field_index]
            ellipse = patches.Ellipse(
                (cx, cy),
                width,
                height,
                linestyle="--",
                edgecolor="black",
                fill=False,
                lw=2,
            )
            ax.add_patch(ellipse)

        # Determine axis labels based on image surface orientation
        cs = self.optic.image_surface.geometry.cs
        if np.any(np.abs(cs.get_effective_rotation_euler())[:2] > 0.01):
            x_label, y_label = "U (mm)", "V (mm)"
        else:
            x_label, y_label = "X (mm)", "Y (mm)"

        ax.axis("square")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_title(f"Hx: {field_coords[0]:.3f}, Hy: {field_coords[1]:.3f}")
        ax.grid(True, alpha=0.25)

    def _finalize_plot(self, fig: Figure, axs: NDArray[np.object_], num_fields: int):
        """Applies final touches to the plot, including a dynamic shared legend.

        Args:
            fig: The Matplotlib figure.
            axs: The array of axes.
            num_fields: The number of fields that were plotted.
        """
        # Remove empty subplot axes
        for i in range(num_fields, len(axs)):
            fig.delaxes(axs[i])

        if num_fields == 0:
            if hasattr(fig, "canvas") and fig.canvas:
                fig.canvas.draw_idle()
            return

        handles, labels = axs[0].get_legend_handles_labels()
        if not handles:
            fig.tight_layout()
            if hasattr(fig, "canvas") and fig.canvas:
                fig.canvas.draw_idle()
            return

        fig.canvas.draw()
        pos_left = axs[0].get_position()
        num_cols = 3

        rightmost_ax_idx = min(num_fields - 1, num_cols - 1)
        pos_right = axs[rightmost_ax_idx].get_position()

        x_center = (pos_left.x0 + pos_right.x1) / 2.0

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(x_center, 0.05),
            ncol=len(self.wavelengths),
        )

        fig.subplots_adjust(bottom=0.2)
