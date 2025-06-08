"""Encircled Energy Analysis

This module provides an encircled energy analysis for optical systems.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.plotting import LegendConfig, Plotter, config  # Updated imports

from .spot_diagram import SpotData, SpotDiagram


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

        if isinstance(wavelength, str):
            if wavelength == "primary":
                processed_wavelength = "primary"
            else:
                raise ValueError(
                    "Invalid wavelength string for EncircledEnergy, only 'primary' "
                    "is supported as a string."
                )
        elif isinstance(wavelength, (float, int)):
            processed_wavelength = float(wavelength)
        else:
            raise TypeError(
                "wavelength argument must be 'primary' or a number (in microns)"
            )

        super().__init__(
            optic,
            fields=fields,
            wavelengths=processed_wavelength,
            num_rings=num_rays,  # Map num_rays to num_rings
            distribution=distribution,
        )

    def view(self, figsize=(7, 4.5), return_fig_ax: bool = False):
        """Plot the Encircled Energy curve.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (7, 4.5).
            return_fig_ax (bool, optional): If True, returns the figure and axes
                objects. Defaults to False, which shows the plot.
        """
        if figsize:
            original_figsize = config.get_config("figure.figsize")
            config.set_config("figure.figsize", figsize)

        fig, ax = None, None
        centered_data = self._center_spots(self.data)
        geometric_size = self.geometric_spot_radius()
        axis_lim = be.max(geometric_size)

        plot_title = f"Wavelength: {self.wavelengths[0]:.4f} µm"
        xlabel = "Radius (mm)"
        ylabel = "Encircled Energy (-)"

        # Prepare legend configuration for the final ax.legend() call.
        # Plotter.plot_line builds legend items from legend_label.
        # We style the legend explicitly at the end using these parameters.
        legend_params_for_ax = LegendConfig(
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            show_legend=True,  # Ensure legend is shown
        )

        for k, field_spot_data_list in enumerate(centered_data):
            field_coords = self.fields[k]
            # EncircledEnergy is for a single wavelength, so field_spot_data_list
            # should contain one SpotData item.
            if not field_spot_data_list:
                continue

            points = field_spot_data_list[0]  # Assuming one SpotData per field

            r_max = axis_lim * 1.2  # buffer
            r_step = be.linspace(0, r_max, self.num_points)

            x_coords = points.x
            y_coords = points.y
            # Bind current_energy_vals and current_radii to default args for closure
            current_energy_vals = points.intensity
            current_radii = be.sqrt(x_coords**2 + y_coords**2)

            def vectorized_ee(r_val, ev=current_energy_vals, rds=current_radii):
                return be.nansum(ev[rds <= r_val])

            ee = be.vectorize(vectorized_ee)(r_step)
            r_np = be.to_numpy(r_step)
            ee_np = be.to_numpy(ee)

            legend_label = f"Hx: {field_coords[0]:.3f}, Hy: {field_coords[1]:.3f}"

            if fig is None:
                fig, ax = Plotter.plot_line(
                    r_np,
                    ee_np,
                    title=plot_title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend_label=legend_label,
                    return_fig_ax=True,
                )
            else:
                Plotter.plot_line(
                    r_np, ee_np, ax=ax, legend_label=legend_label, return_fig_ax=True
                )

        if ax:  # Final styling if plot was created
            ax.set_xlim((0, None))
            ax.set_ylim((0, None))
            # Apply collected legend configuration
            # Plotter.plot_line with legend_label adds to legend items.
            # Now, style the legend itself.
            final_legend_kwargs = {
                "bbox_to_anchor": legend_params_for_ax.get(
                    "bbox_to_anchor", config.get_config("legend.bbox_to_anchor")
                ),
                "loc": legend_params_for_ax.get(
                    "legend_loc", config.get_config("legend.loc")
                ),
                "frameon": config.get_config("legend.frameon"),  # Use global for these
                "shadow": config.get_config("legend.shadow"),
                "fancybox": config.get_config("legend.fancybox"),
                "ncol": config.get_config("legend.ncol"),
                "fontsize": config.get_config("font.size_legend"),
            }

            should_show_legend = legend_params_for_ax.get(
                "show_legend", config.get_config("legend.show")
            )
            handles, labels = ax.get_legend_handles_labels()
            if should_show_legend and handles:  # Check if handles is not empty
                ax.legend(
                    **{k: v for k, v in final_legend_kwargs.items() if v is not None}
                )

            # fig.tight_layout() is usually handled by Plotter/plt.show().
            # Call before _handle_fig_ax_return_logic for specific needs.

        if figsize:
            config.set_config("figure.figsize", original_figsize)

        return Plotter.finalize_plot_objects(return_fig_ax, fig, ax)

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

    # _plot_field method is now integrated into view()

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
