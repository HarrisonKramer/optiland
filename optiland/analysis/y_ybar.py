"""Y Y-bar Analysis

This module provides a y y-bar analysis for optical systems.
This is a plot of the marginal ray height versus the chief ray height
for each surface in the system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.plotting import LegendConfig, Plotter, config, themes  # Updated imports

from .base import BaseAnalysis


class YYbar(BaseAnalysis):
    """Class representing the YYbar analysis of an optic.

    Args:
        optic (Optic): The optic object to analyze.
        wavelength (str, optional): The wavelength to use for analysis.
            Defaults to 'primary'.

    Methods:
        view(figsize=(7, 5.5)): Visualizes the YYbar analysis.

    """

    def __init__(self, optic, wavelength="primary"):
        if isinstance(wavelength, str) and wavelength == "primary":
            processed_wavelength = "primary"
        elif isinstance(wavelength, (float, int)):
            processed_wavelength = wavelength
        else:
            raise TypeError(
                f"Unsupported wavelength type for YYbar: {type(wavelength)}"
            )

        super().__init__(optic, wavelengths=processed_wavelength)

    def _generate_data(self):
        ya, _ = self.optic.paraxial.marginal_ray()
        yb, _ = self.optic.paraxial.chief_ray()

        return {"ya": ya.flatten(), "yb": yb.flatten()}

    def view(self, figsize=(7, 5.5), return_fig_ax: bool = False):
        """Visualizes the Y-Ybar diagram using Plotter.

        Args:
            figsize (tuple): The size of the figure (width, height).
                Defaults to (7, 5.5).
            return_fig_ax (bool, optional): If True, returns the figure and axes
                objects. Defaults to False, which shows the plot.
        """
        if figsize:
            original_figsize = config.get_config("figure.figsize")
            config.set_config("figure.figsize", figsize)

        fig, ax = None, None
        ya = self.data["ya"]
        yb = self.data["yb"]

        plot_title = "Y-Ybar Diagram"
        xlabel = "Chief Ray Height (mm)"
        ylabel = "Marginal Ray Height (mm)"

        active_theme = themes.get_active_theme_dict()
        axis_line_color = active_theme.get(
            "grid.color", "k"
        )  # Use grid color for axis lines

        for k in range(2, len(ya)):
            legend_label = None  # Plotter may show legend if global show_legend is True
            if k == 2:
                legend_label = "Surface 1"
            elif k == len(ya) - 1:
                legend_label = "Image"

            x_segment = [be.to_numpy(yb[k - 1]), be.to_numpy(yb[k])]
            y_segment = [be.to_numpy(ya[k - 1]), be.to_numpy(ya[k])]

            plot_kwargs = {
                "marker": ".",
                "linestyle": "-",
                "markersize": 8,
                "legend_label": legend_label,
                "return_fig_ax": True,
            }

            if fig is None:
                fig, ax = Plotter.plot_line(
                    x_segment,
                    y_segment,
                    title=plot_title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    **plot_kwargs,
                )
            else:
                Plotter.plot_line(x_segment, y_segment, ax=ax, **plot_kwargs)

        if ax:
            ax.axhline(y=0, linewidth=0.5, color=axis_line_color)
            ax.axvline(x=0, linewidth=0.5, color=axis_line_color)

            # Ensure legend is shown if there are labels
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # Only show legend if there are labeled items
                legend_cfg = LegendConfig(
                    show_legend=True
                )  # Minimal config to ensure it's shown
                ax.legend(
                    loc=legend_cfg.get("legend_loc", config.get_config("legend.loc")),
                    frameon=legend_cfg.get(
                        "legend_frameon", config.get_config("legend.frameon")
                    ),
                    shadow=legend_cfg.get(
                        "legend_shadow", config.get_config("legend.shadow")
                    ),
                    fancybox=legend_cfg.get(
                        "legend_fancybox", config.get_config("legend.fancybox")
                    ),
                    ncol=legend_cfg.get(
                        "legend_ncol", config.get_config("legend.ncol")
                    ),
                    fontsize=config.get_config("font.size_legend"),
                )

        if figsize:
            config.set_config("figure.figsize", original_figsize)

        return Plotter.finalize_plot_objects(return_fig_ax, fig, ax)
