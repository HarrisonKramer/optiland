"""Ray Aberration Fan Analysis

This module provides a ray fan analysis for optical systems.

Kramer Harrison, 2024
"""

from matplotlib.lines import Line2D  # For custom legend

import optiland.backend as be
from optiland.plotting import Plotter  # Updated imports

from .base import BaseAnalysis


class RayFan(BaseAnalysis):
    """Represents a ray fan aberration analysis for an optic.

    Args:
        optic (Optic): The optic object to analyze.
        fields (str or list, optional): The fields to analyze.
            Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points in the ray fan.
            Defaults to 256.

    Attributes:
        optic (Optic): The optic object being analyzed.
        fields (list): The fields being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of points in the ray fan.
        data (dict): The generated ray fan data.

    Methods:
        view(figsize=(10, 3.33)): Displays the ray fan plot.

    """

    def __init__(self, optic, fields="all", wavelengths="all", num_points=256):
        _optic_ref = optic
        if fields == "all":
            self.fields = _optic_ref.fields.get_field_coords()
        else:
            self.fields = fields

        if num_points % 2 == 0:
            self.num_points = num_points + 1  # force to be odd so a point lies at P=0
        else:
            self.num_points = num_points

        super().__init__(optic, wavelengths)

    def view(self, figsize=(10, 3.33), return_fig_ax: bool = False):
        """Displays the ray fan plot using Plotter.

        Args:
            figsize (tuple, optional): The base size for each field's row of subplots.
                The total figure height will be `figsize[1] * len(self.fields)`.
                Defaults to (10, 3.33).
            return_fig_ax (bool, optional): If True, returns the figure and axes
                objects. Defaults to False, which shows the plot.
        """
        plot_callbacks = []
        num_fields = len(self.fields)
        calculated_figsize = (figsize[0], figsize[1] * num_fields)

        Px_all = self.data["Px"]
        Py_all = self.data["Py"]

        # Define colors for wavelengths for the manual figure legend.
        # Use current theme's property cycler or a fallback list.
        from optiland.plotting import themes

        active_theme_settings = themes.get_active_theme_dict()
        prop_cycle = active_theme_settings.get("axes.prop_cycle")

        legend_line_colors = []
        if prop_cycle:
            for item in prop_cycle:
                if "color" in item:
                    legend_line_colors.append(item["color"])

        if not legend_line_colors:  # Fallback if no prop_cycle or colors
            legend_line_colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]  # Matplotlib v2.0 defaults

        wavelength_colors = {
            wav: legend_line_colors[i % len(legend_line_colors)]
            for i, wav in enumerate(self.wavelengths)
        }

        for _field_idx, field_val in enumerate(self.fields):
            # Tangential callback (Py vs ey)
            def tangential_callback(ax, plot_idx, field=field_val):
                for _wav_idx, wavelength in enumerate(self.wavelengths):
                    ey_data = self.data[f"{field}"][f"{wavelength}"]["y"]
                    intensity_y = self.data[f"{field}"][f"{wavelength}"]["intensity_y"]
                    ey_data[intensity_y == 0] = be.nan

                    Py_np = be.to_numpy(Py_all)
                    ey_np = be.to_numpy(ey_data)

                    Plotter.plot_line(
                        Py_np,
                        ey_np,
                        ax=ax,
                        legend_label=f"{wavelength:.4f} µm",
                        return_fig_ax=True,
                        color=wavelength_colors[wavelength],
                    )
                # _apply_ax_styling (via Plotter.plot_line) handles general styling.
                # Add specific markings like hlines/vlines here.
                ax.axhline(
                    y=0, lw=1, color="gray"
                )  # Keep these as they are specific markings
                ax.axvline(x=0, lw=1, color="gray")
                ax.set_xlabel("$P_y$")
                ax.set_ylabel("$\\epsilon_y$ (mm)")
                ax.set_xlim((-1, 1))
                ax.set_title(f"Tangential - Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

            plot_callbacks.append(tangential_callback)

            # Sagittal callback (Px vs ex)
            def sagittal_callback(ax, plot_idx, field=field_val):
                for _wav_idx, wavelength in enumerate(self.wavelengths):
                    ex_data = self.data[f"{field}"][f"{wavelength}"]["x"]
                    intensity_x = self.data[f"{field}"][f"{wavelength}"]["intensity_x"]
                    ex_data[intensity_x == 0] = be.nan

                    Px_np = be.to_numpy(Px_all)
                    ex_np = be.to_numpy(ex_data)

                    Plotter.plot_line(
                        Px_np,
                        ex_np,
                        ax=ax,
                        legend_label=f"{wavelength:.4f} µm",
                        return_fig_ax=True,
                        color=wavelength_colors[wavelength],
                    )
                # Add specific markings
                ax.axhline(y=0, lw=1, color="gray")  # Keep these specific markings
                ax.axvline(x=0, lw=1, color="gray")
                ax.set_xlabel("$P_x$")
                ax.set_ylabel("$\\epsilon_x$ (mm)")
                ax.set_xlim((-1, 1))  # xlim should be on x-axis
                ax.set_title(f"Sagittal - Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

            plot_callbacks.append(sagittal_callback)

        fig, axs = Plotter.plot_subplots(
            num_rows=num_fields,
            num_cols=2,
            plot_callbacks=plot_callbacks,
            sharex=True,
            sharey=True,
            # main_title="Ray Fan Plot", # Optional: original did not have one
            return_fig_ax=True,  # Get fig and axs back
            figsize=calculated_figsize,  # Pass figsize directly
        )

        if fig is not None and axs is not None:  # Ensure fig and axs were returned
            # Create custom figure-level legend
            legend_handles = [
                Line2D([0], [0], color=wavelength_colors[wav], lw=2)
                for wav in self.wavelengths
            ]
            legend_labels = [f"{wav:.4f} µm" for wav in self.wavelengths]

            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",  # As per original
                bbox_to_anchor=(-0.1, -0.2),  # As per original, may need adjustment
                ncol=3,  # As per original
            )

            # Original had plt.subplots_adjust(top=1)
            # Plotter.plot_subplots uses fig.tight_layout().
            # If specific adjustment is still needed, it can be done here.
            # For example: fig.subplots_adjust(bottom=0.2) to make space for legend.
            # The bbox_to_anchor might mean the legend is outside the main plot area.
            # Default tight_layout might not account for fig.legend well.
            # A common adjustment for legends below the plot:
            fig.tight_layout(rect=[0, 0.1, 1, 0.95])  # rect=[left, bottom, right, top]

        # Final plot handling using Plotter's logic
        return Plotter.finalize_plot_objects(return_fig_ax, fig, axs)

    def _generate_data(self):
        """Generates the ray fan data.

        Returns:
            dict: The generated ray fan data.

        """
        data = {}
        data["Px"] = be.linspace(-1, 1, self.num_points)
        data["Py"] = be.linspace(-1, 1, self.num_points)
        for field in self.fields:
            Hx = field[0]
            Hy = field[1]

            data[f"{field}"] = {}
            for wavelength in self.wavelengths:
                data[f"{field}"][f"{wavelength}"] = {}

                self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_x",
                )
                data[f"{field}"][f"{wavelength}"]["x"] = self.optic.surface_group.x[
                    -1, :
                ]
                data[f"{field}"][f"{wavelength}"]["intensity_x"] = (
                    self.optic.surface_group.intensity[-1, :]
                )

                self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_y",
                )
                data[f"{field}"][f"{wavelength}"]["y"] = self.optic.surface_group.y[
                    -1, :
                ]
                data[f"{field}"][f"{wavelength}"]["intensity_y"] = (
                    self.optic.surface_group.intensity[-1, :]
                )

        # remove distortion
        wave_ref = self.optic.primary_wavelength
        for field in self.fields:
            x_offset = data[f"{field}"][f"{wave_ref}"]["x"][self.num_points // 2]
            y_offset = data[f"{field}"][f"{wave_ref}"]["y"][self.num_points // 2]
            for wavelength in self.wavelengths:
                orig_x = data[f"{field}"][f"{wavelength}"]["x"]
                orig_y = data[f"{field}"][f"{wavelength}"]["y"]
                data[f"{field}"][f"{wavelength}"]["x"] = orig_x - x_offset
                data[f"{field}"][f"{wavelength}"]["y"] = orig_y - y_offset

        return data
