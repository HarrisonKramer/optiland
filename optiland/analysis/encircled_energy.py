"""Encircled Energy Analysis

This module provides an encircled energy analysis for optical systems.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.analysis.spot_diagram import SpotDiagram
from optiland.plotting.core import Plotter

from .spot_diagram import SpotData


class EncircledEnergy(SpotDiagram):
    """Class representing the Encircled Energy analysis of a given optic.

    Args:
        optic (Optic): The optic for which the Encircled Energy analysis is
            performed.
        fields (str or tuple, optional): The fields for which the analysis is
            performed. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength at which the
            analysis is performed. Defaults to 'primary'.
        num_rays (int, optional): The number of rays used for the analysis.
            Defaults to 100000.
        distribution (str, optional): The distribution of rays.
            Defaults to 'random'.
        num_points (int, optional): The number of points used for plotting the
            Encircled Energy curve. Defaults to 256.

    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelength="primary",
        num_rays=100_000,
        distribution="random",
        num_points=256,
    ):
        self.num_points = num_points
        if wavelength == "primary":
            wavelength = optic.primary_wavelength

        super().__init__(optic, fields, [wavelength], num_rays, distribution)

    def view(self, figsize=(7, 4.5)):
        """Plot the Encircled Energy curve.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (7, 4.5).

        """
        plotter = Plotter()
        fig, ax = plotter.create_figure_and_axes(figsize=figsize)

        data = self._center_spots(self.data)
        geometric_size = self.geometric_spot_radius()
        axis_lim = be.max(geometric_size)
        for k, field_data in enumerate(data):
            # Pass plotter and color index k to _plot_field
            self._plot_field(ax, plotter, k, field_data, self.fields[k], axis_lim, self.num_points)

        plotter.add_legend(ax, bbox_to_anchor=(1.05, 0.5), loc="center left")
        plotter.set_labels(ax, xlabel="Radius (mm)", ylabel="Encircled Energy (-)")
        plotter.set_title(ax, title=f"Wavelength: {self.wavelengths[0]:.4f} µm")

        # Get current y_max before setting xlim, as xlim might affect auto-scaled y_max in some backends
        current_y_max = ax.get_ylim()[1]
        plotter.set_xlim(ax, (0, axis_lim * 1.2)) # Use calculated axis_lim for xlim
        plotter.set_ylim(ax, (0, current_y_max)) # Set ylim from 0 to current auto-scaled max or a bit more

        plotter.tight_layout(fig)
        plotter.show_plot()

    def centroid(self):
        """Calculate the centroid of the Encircled Energy.

        Returns:
            list: A list of tuples representing the centroid coordinates for
                each field.

        """
        centroid = []
        for field_data in self.data:
            spot_data_item = field_data[0]
            centroid_x = be.mean(spot_data_item.x)
            centroid_y = be.mean(spot_data_item.y)
            centroid.append((centroid_x, centroid_y))
        return centroid

    def _plot_field(self, ax, plotter, color_idx, field_data, field, axis_lim, num_points, buffer=1.2):
        """Plot the Encircled Energy curve for a specific field.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot the curve.
            plotter: The Plotter instance for styling.
            color_idx (int): Index for selecting line color.
            field_data (list): List of field data.
            field (tuple): Tuple representing the normalized field coordinates.
            axis_lim (float): Maximum axis limit.
            num_points (int): Number of points for plotting the curve.
            buffer (float, optional): Buffer factor for the axis limit.
                Defaults to 1.2.

        """
        r_max = axis_lim * buffer
        r_step = be.linspace(0, r_max, num_points)

            line_colors = plotter.config.get_style('line_colors')
            # Use color_idx for consistency, it represents the field index k from the view method
            line_color = line_colors[color_idx % len(line_colors)]

            for points in field_data: # This loop is over wavelengths, but EncircledEnergy is monochromatic by default from init
            x = points.x
            y = points.y
            # energy and intensity are used interchangeably here
            energy = points.intensity
            radii = be.sqrt(x**2 + y**2)

            def vectorized_ee(r):
                return be.nansum(energy[radii <= r])  # noqa: B023

            # element‑wise encircled energy (Tensor)
            ee = be.vectorize(vectorized_ee)(r_step)

            # convert both to plain numpy for plotting
            r_np = be.to_numpy(r_step)
            ee_np = be.to_numpy(ee)
                ax.plot(r_np, ee_np, color=line_color, label=f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

    def _generate_field_data(
        self,
        field,
        wavelength,
        num_rays=100,
        distribution="hexapolar",
        coordinates="local",
    ):
        """Generate the field data for a specific field and wavelength.

        Args:
            field (tuple): Tuple representing the field coordinates.
            wavelength (float): The wavelength.
            num_rays (int, optional): The number of rays. Defaults to 100.
            distribution (str, optional): The distribution of rays.
                Defaults to 'hexapolar'.
            coordinates (str): Coordinate system choice (ignored).

        Returns:
            SpotData: SpotData object containing x, y, and intensity arrays.

        """
        self.optic.trace(*field, wavelength, num_rays, distribution)
        x = self.optic.surface_group.x[-1, :]
        y = self.optic.surface_group.y[-1, :]
        intensity = self.optic.surface_group.intensity[-1, :]
        return SpotData(x=x, y=y, intensity=intensity)
