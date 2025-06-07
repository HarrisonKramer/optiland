"""Distortion Analysis

This module provides a distortion analysis for optical systems.

Kramer Harrison, 2024
"""

import numpy as np

import optiland.backend as be
from optiland.plotting import LegendConfig, Plotter, config  # Updated imports

from .base import BaseAnalysis


class Distortion(BaseAnalysis):
    """Represents a distortion analysis for an optic.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points to generate for the
            analysis. Defaults to 128.
        distortion_type (str, optional): The type of distortion analysis.
            Defaults to 'f-tan'.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of points generated for the analysis.
        distortion_type (str): The type of distortion analysis.
        data (list): The generated distortion data.

    Methods:
        view(figsize=(7, 5.5)): Visualizes the distortion analysis.

    """

    def __init__(
        self,
        optic,
        wavelengths="all",
        num_points=128,
        distortion_type="f-tan",
    ):
        self.num_points = num_points
        self.distortion_type = distortion_type
        super().__init__(optic, wavelengths)

    def view(self, figsize=(7, 5.5), return_fig_ax: bool = False):
        """Visualize the distortion analysis.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).
            return_fig_ax (bool, optional): If True, returns the figure and axes
                objects. Defaults to False, which shows the plot.
        """
        if figsize:
            original_figsize = config.get_config("figure.figsize")
            config.set_config("figure.figsize", figsize)

        fig, ax = None, None
        field = be.linspace(1e-10, self.optic.fields.max_field, self.num_points)
        field_np = be.to_numpy(field)

        legend_labels_for_plotter = []

        for k, wavelength in enumerate(self.wavelengths):
            dist_k = be.to_numpy(self.data[k])
            legend_label = f"{wavelength:.4f} µm"
            legend_labels_for_plotter.append(legend_label)

            if fig is None and ax is None:  # First plot call
                fig, ax = Plotter.plot_line(
                    dist_k,
                    field_np,
                    xlabel="Distortion (%)",
                    ylabel="Field",
                    legend_label=legend_label,
                    return_fig_ax=True,  # Always true to get fig, ax
                )
            else:  # Subsequent plots on the same axes
                Plotter.plot_line(
                    dist_k,
                    field_np,
                    ax=ax,
                    legend_label=legend_label,
                    return_fig_ax=True,  # Keep getting ax to ensure it's not None
                )

        if ax:  # Ensure ax is not None before styling
            ax.axvline(x=0, color="k", linewidth=1, linestyle="--")

            # Symmetrize xlim
            current_xlim = ax.get_xlim()
            max_abs_xlim = max(np.abs(current_xlim[0]), np.abs(current_xlim[1]))
            ax.set_xlim([-max_abs_xlim, max_abs_xlim])
            ax.set_ylim([0, None])

            # Apply legend using Plotter's default logic or a specific LegendConfig.
            # Plotter.plot_line adds labels if legend_label is provided.
            # We need to ensure the legend is displayed.
            # legend_config can be used for more specific styling.
            # For now, relying on plot_line's behavior and global config.
            # If a legend object is needed for further manipulation:
            if ax.get_legend_handles_labels()[0]:  # Check if there are any labels
                # Create LegendConfig for specific styling if needed.
                legend_config_params = LegendConfig(
                    show_legend=True,
                    legend_bbox_to_anchor=(1.05, 0.5),
                    legend_loc="center left",
                )

                # Legend is built by Plotter.plot_line.
                # Style it here if Plotter didn't fully handle it.
                # Retrieve current legend settings from config for robustness.

                final_legend_config = {
                    "title": legend_config_params.get(
                        "legend_title", config.get_config("legend.title")
                    ),
                    "loc": legend_config_params.get(
                        "legend_loc", config.get_config("legend.loc")
                    ),
                    "frameon": legend_config_params.get(
                        "legend_frameon", config.get_config("legend.frameon")
                    ),
                    "shadow": legend_config_params.get(
                        "legend_shadow", config.get_config("legend.shadow")
                    ),
                    "fancybox": legend_config_params.get(
                        "legend_fancybox", config.get_config("legend.fancybox")
                    ),
                    "ncol": legend_config_params.get(
                        "legend_ncol", config.get_config("legend.ncol")
                    ),
                    "bbox_to_anchor": legend_config_params.get(
                        "legend_bbox_to_anchor",
                        config.get_config("legend.bbox_to_anchor"),
                    ),
                    "fontsize": config.get_config("font.size_legend"),
                }
                ax.legend(**final_legend_config)

        if figsize:  # Reset figsize to original if it was changed
            config.set_config("figure.figsize", original_figsize)

        # Final plot handling
        return Plotter.finalize_plot_objects(return_fig_ax, fig, ax)

    def _generate_data(self):
        """Generate data for analysis.

        This method generates the distortion data to be used for plotting.

        Returns:
            list: A list of distortion data points.

        """
        Hx = be.zeros(self.num_points)
        Hy = be.linspace(1e-10, 1, self.num_points)

        data = []
        for wavelength in self.wavelengths:
            self.optic.trace_generic(Hx=Hx, Hy=Hy, Px=0, Py=0, wavelength=wavelength)
            yr = self.optic.surface_group.y[-1, :]

            const = yr[0] / (be.tan(1e-10 * be.radians(self.optic.fields.max_field)))

            if self.distortion_type == "f-tan":
                yp = const * be.tan(Hy * be.radians(self.optic.fields.max_field))
            elif self.distortion_type == "f-theta":
                yp = const * Hy * be.radians(self.optic.fields.max_field)
            else:
                raise ValueError(
                    '''Distortion type must be "f-tan" or
                                 "f-theta"'''
                )

            data.append(100 * (yr - yp) / yp)

        return data
