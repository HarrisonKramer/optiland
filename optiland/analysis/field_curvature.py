"""Field Curvature Analysis

This module provides a field curvature analysis for optical systems.

Kramer Harrison, 2024
"""

import numpy as np

import optiland.backend as be
from optiland.plotting import LegendConfig, Plotter, config  # Updated imports

from .base import BaseAnalysis


class FieldCurvature(BaseAnalysis):
    """Represents a class for analyzing field curvature of an optic.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points to generate for the
            analysis. Defaults to 128.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of points generated for the analysis.
        data (list): The generated data for the analysis.

    Methods:
        view(figsize=(8, 5.5)): Displays a plot of the field curvature
            analysis.

    """

    def __init__(self, optic, wavelengths="all", num_points=128):
        self.num_points = num_points
        super().__init__(optic, wavelengths)

    def view(self, figsize=(8, 5.5), return_fig_ax: bool = False):
        """Displays a plot of the field curvature analysis.

        Args:
            figsize (tuple, optional): The size of the figure. Defaults to (8, 5.5).
            return_fig_ax (bool, optional): If True, returns the figure and axes
                objects. Defaults to False, which shows the plot.
        """
        if figsize:
            original_figsize = config.get_config("figure.figsize")
            config.set_config("figure.figsize", figsize)

        fig, ax = None, None
        field_np = be.to_numpy(
            be.linspace(0, self.optic.fields.max_field, self.num_points)
        )

        plot_title = "Field Curvature"
        xlabel = "Image Plane Delta (mm)"
        ylabel = "Field"

        # Colors are cycled by Plotter. Linestyles distinguish T & S.
        # To ensure T & S for the same wavelength use the same color base (as C{k}
        # and C{k}-- did), we explicitly pass the color to Plotter.plot_line.

        # Get theme colors for consistent color per wavelength
        from optiland.plotting import themes

        active_theme = themes.get_active_theme_dict()
        prop_cycle = active_theme.get("axes.prop_cycle", None)
        default_colors = [
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
        ]
        colors = (
            [item["color"] for item in prop_cycle] if prop_cycle else default_colors
        )

        for k, wavelength in enumerate(self.wavelengths):
            dk_np_tan = be.to_numpy(self.data[k][0])
            dk_np_sag = be.to_numpy(self.data[k][1])
            current_color = colors[k % len(colors)]

            if fig is None:  # First plot
                fig, ax = Plotter.plot_line(
                    dk_np_tan,
                    field_np,
                    title=plot_title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend_label=f"{wavelength:.4f} µm, Tangential",
                    return_fig_ax=True,
                    color=current_color,  # Explicit color
                    linestyle="-",
                )
                Plotter.plot_line(  # Sagittal for the first wavelength
                    dk_np_sag,
                    field_np,
                    ax=ax,
                    legend_label=f"{wavelength:.4f} µm, Sagittal",
                    return_fig_ax=True,
                    color=current_color,  # Same color
                    linestyle="--",
                )
            else:  # Subsequent wavelengths
                Plotter.plot_line(
                    dk_np_tan,
                    field_np,
                    ax=ax,
                    legend_label=f"{wavelength:.4f} µm, Tangential",
                    return_fig_ax=True,
                    color=current_color,
                    linestyle="-",
                )
                Plotter.plot_line(
                    dk_np_sag,
                    field_np,
                    ax=ax,
                    legend_label=f"{wavelength:.4f} µm, Sagittal",
                    return_fig_ax=True,
                    color=current_color,
                    linestyle="--",
                )

        if ax:
            ax.set_ylim([0, self.optic.fields.max_field])
            current_xlim = ax.get_xlim()
            max_abs_xlim = max(np.abs(current_xlim[0]), np.abs(current_xlim[1]))
            ax.set_xlim([-max_abs_xlim, max_abs_xlim])

            ax.axvline(x=0, color=active_theme.get("grid.color", "k"), linewidth=0.5)

            legend_cfg = LegendConfig(
                bbox_to_anchor=(1.05, 0.5), loc="center left", show_legend=True
            )
            show_legend_param = legend_cfg.get("show_legend")
            should_show_legend = (
                show_legend_param
                if show_legend_param is not None
                else config.get_config("legend.show")
            )
            handles, _ = ax.get_legend_handles_labels()  # Use _ for unused labels var

            if should_show_legend and handles:  # Check if handles is not empty
                ax.legend(
                    bbox_to_anchor=legend_cfg.get("bbox_to_anchor"),
                    loc=legend_cfg.get("loc"),
                    frameon=config.get_config("legend.frameon"),
                    shadow=config.get_config("legend.shadow"),
                    fancybox=config.get_config("legend.fancybox"),
                    ncol=config.get_config("legend.ncol"),
                    fontsize=config.get_config("font.size_legend"),
                )
            # fig.tight_layout() is often handled by Plotter/plt.show().
            # For bbox_to_anchor legends, tight_layout might need rect param.
            # Plotter's default handling is usually sufficient.

        if figsize:
            config.set_config("figure.figsize", original_figsize)

        return Plotter.finalize_plot_objects(return_fig_ax, fig, ax)

    def _generate_data(self):
        """Generates field curvature data for each wavelength by calculating the
            tangential and sagittal intersections.

        Returns:
            list: A list of np.ndarry containing the tangential and sagittal
                intersection points for each wavelength.

        """
        data = []
        for wavelength in self.wavelengths:
            tangential = self._intersection_parabasal_tangential(wavelength)
            sagittal = self._intersection_parabasal_sagittal(wavelength)

            data.append([tangential, sagittal])

        return data

    def _intersection_parabasal_tangential(self, wavelength, delta=1e-5):
        """Calculate the intersection of parabasal rays in tangential plane.

        Args:
            wavelength (float): The wavelength of the light.
            delta (float, optional): The delta value in normalized pupil y
                coordinates for pairs of parabasal rays. Defaults to 1e-5.

        Returns:
            numpy.ndarray: The calculated intersection values.

        """
        Hx = be.zeros(2 * self.num_points)
        Hy = be.repeat(be.linspace(0, 1, self.num_points), 2)

        Px = be.zeros(2 * self.num_points)
        Py = be.tile(be.array([-delta, delta]), self.num_points)

        self.optic.trace_generic(Hx, Hy, Px, Py, wavelength=wavelength)

        M1 = self.optic.surface_group.M[-1, ::2]
        N1 = self.optic.surface_group.N[-1, ::2]

        M2 = self.optic.surface_group.M[-1, 1::2]
        N2 = self.optic.surface_group.N[-1, 1::2]

        y01 = self.optic.surface_group.y[-1, ::2]
        z01 = self.optic.surface_group.z[-1, ::2]

        y02 = self.optic.surface_group.y[-1, 1::2]
        z02 = self.optic.surface_group.z[-1, 1::2]

        t1 = (M2 * z01 - M2 * z02 - N2 * y01 + N2 * y02) / (M1 * N2 - M2 * N1)

        return t1 * N1

    def _intersection_parabasal_sagittal(self, wavelength, delta=1e-5):
        """Calculate the intersection of parabasal rays in sagittal plane.

        Args:
            wavelength (float): The wavelength of the light.
            delta (float, optional): The delta value in normalized pupil y
                coordinates for pairs of parabasal rays. Defaults to 1e-5.

        Returns:
            numpy.ndarray: The calculated intersection values.

        """
        Hx = be.zeros(2 * self.num_points)
        Hy = be.repeat(be.linspace(0, 1, self.num_points), 2)

        Px = be.tile(be.array([-delta, delta]), self.num_points)
        Py = be.zeros(2 * self.num_points)

        self.optic.trace_generic(Hx, Hy, Px, Py, wavelength=wavelength)

        L1 = self.optic.surface_group.L[-1, ::2]
        N1 = self.optic.surface_group.N[-1, ::2]

        L2 = self.optic.surface_group.L[-1, 1::2]
        N2 = self.optic.surface_group.N[-1, 1::2]

        x01 = self.optic.surface_group.x[-1, ::2]
        z01 = self.optic.surface_group.z[-1, ::2]

        x02 = self.optic.surface_group.x[-1, 1::2]
        z02 = self.optic.surface_group.z[-1, 1::2]

        t2 = (L2 * z01 - L2 * z02 - N2 * x01 + N2 * x02) / (L1 * N2 - L2 * N1)

        return t2 * N1
