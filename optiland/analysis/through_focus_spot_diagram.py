"""Through Focus Spot Diagram Analysis

This module provides a class for performing through-focus spot diagram
analysis, calculating the spot diagram at various focal planes.

Kramer Harrison, 2025
"""

import functools  # For partial
from typing import Literal

import numpy as np
from matplotlib.lines import Line2D  # For legend proxy artists if needed

import optiland.backend as be
from optiland.analysis.spot_diagram import SpotDiagram  # SpotDiagram.SpotData is used
from optiland.analysis.through_focus import ThroughFocusAnalysis
from optiland.plotting import Plotter, themes


class ThroughFocusSpotDiagram(ThroughFocusAnalysis):
    """Performs spot diagram analysis over a range of focal planes.

    This class extends `ThroughFocusAnalysis` to specifically calculate and
    report RMS spot radii from spot diagrams at various focal positions.
    It utilizes the `SpotDiagram` class for the core calculations at each
    focal plane.

    Attributes:
        optic (optiland.optic.Optic): The optical system being analyzed.
        delta_focus (float): The focal shift increment in mm.
        num_steps (int): Number of focal planes analyzed before and after
            the nominal focus.
        fields (list): Resolved list of field coordinates for analysis.
        wavelengths (list): Resolved list of wavelengths for analysis.
        num_rings (int): Number of rings for pupil sampling in the
            `SpotDiagram` calculation.
        distribution (str): Pupil sampling distribution type (e.g.,
            'hexapolar', 'random') for `SpotDiagram`.
        coordinates (Literal["global", "local"]): Coordinate system used for
            spot data generation within `SpotDiagram`.
        results (list[dict[float, list[float]]]): A list where each item is a
            dictionary. Each dictionary corresponds to a single focal plane
            and maps the delta focus (float, in mm) to a list of RMS spot
            radii (list of floats, in mm). Each RMS spot radius in the list
            corresponds to a field defined in `self.fields`, calculated at the
            primary wavelength.
    """

    def __init__(
        self,
        optic,
        delta_focus: float = 0.1,
        num_steps: int = 5,
        fields="all",
        wavelengths="all",
        num_rings: int = 6,
        distribution: str = "hexapolar",
        coordinates: Literal["global", "local"] = "local",
    ):
        """Initializes the ThroughFocusSpotDiagram analysis.

        Args:
            optic (optiland.optic.Optic): The optical system to analyze.
            delta_focus (float, optional): The increment of focal shift in mm.
                Defaults to 0.1.
            num_steps (int, optional): The number of focal planes to analyze
                on either side of the nominal focus. Defaults to 5. Must be in
                range [1, 7].
            fields (list[tuple[float,float]] | str, optional): Fields for
                analysis. If "all", uses all fields from `optic.fields`.
                Otherwise, expects a list of field coordinates.
                Defaults to "all".
            wavelengths (list[float] | str, optional): Wavelengths for
                analysis. If "all", uses all wavelengths from
                `optic.wavelengths`. Otherwise, expects a list of
                wavelength values. Defaults to "all".
            num_rings (int, optional): Number of rings for pupil sampling in
                the `SpotDiagram` calculation. Defaults to 6.
            distribution (str, optional): Pupil sampling distribution type for
                `SpotDiagram` (e.g., 'hexapolar', 'random').
                Defaults to "hexapolar".
            coordinates (Literal["global", "local"], optional): Coordinate
                system for spot data generation in `SpotDiagram`.
                Defaults to "local".
        """
        self.num_rings = num_rings
        self.distribution = distribution
        if coordinates not in ["global", "local"]:
            raise ValueError("Coordinates must be 'global' or 'local'.")
        self.coordinates = coordinates

        super().__init__(
            optic,
            delta_focus=delta_focus,
            num_steps=num_steps,
            fields=fields,
            wavelengths=wavelengths,
        )

    def _perform_analysis_at_focus(self):
        """Calculates spot diagram data at the current focal plane."""
        spot_diagram_at_focus = SpotDiagram(
            self.optic,
            fields=self.fields,
            wavelengths=self.wavelengths,
            num_rings=self.num_rings,
            distribution=self.distribution,
            coordinates=self.coordinates,
        )
        return spot_diagram_at_focus.data

    def _render_single_spot_diagram_subplot(
        self,
        ax,
        field_coord,
        position_val,
        spot_data_list_all_wl,
        row_idx,
        col_idx,
        num_total_rows,
        x_label,
        y_label,
        global_axis_limit,
        theme_settings,
        active_theme_colors,
    ):
        """Renders a single spot diagram subplot using Plotter methods."""
        defocus_offset = float(position_val) - be.to_numpy(self.nominal_focus).item()
        centroid_x, centroid_y = self._get_spot_centroid(spot_data_list_all_wl)

        markers = ["o", "s", "^"]
        for k_wl, spot_data_item in enumerate(spot_data_list_all_wl):
            x = be.to_numpy(spot_data_item.x - centroid_x)
            y = be.to_numpy(spot_data_item.y - centroid_y)
            i_mask = be.to_numpy(spot_data_item.intensity) != 0

            if np.any(i_mask):
                # Use themed color for this wavelength for legend consistency.
                color = active_theme_colors[k_wl % len(active_theme_colors)]
                Plotter.plot_scatter(
                    x[i_mask],
                    y[i_mask],
                    ax=ax,
                    s=10,
                    marker=markers[k_wl % len(markers)],
                    alpha=0.7,
                    color=color,  # Explicit color for consistency
                    return_fig_ax=True,
                )

        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle=":", alpha=theme_settings.get("grid.alpha", 0.25))

        title = f"Field: ({field_coord[0]:.2f},{field_coord[1]:.2f})"
        if row_idx == 0:
            title = f"Defocus: {defocus_offset:+.3f} mm\n{title}"
        ax.set_title(title, fontsize=10)

        if row_idx == num_total_rows - 1:
            ax.set_xlabel(x_label)
        if col_idx == 0:
            ax.set_ylabel(y_label)

        ax.set_xlim(-global_axis_limit, global_axis_limit)
        ax.set_ylim(-global_axis_limit, global_axis_limit)

    def view(self, figsize_per_plot=(3, 3), buffer=1.05, return_fig_ax: bool = False):
        """Visualizes through-focus spot diagrams using Plotter.

        Rows represent defocus positions, columns represent fields. Each subplot
        shows spot diagrams for all analyzed wavelengths.

        Args:
            figsize_per_plot (tuple): Approx size (W,H) for each subplot.
            buffer (float): Buffer for axis limits.
            return_fig_ax (bool): If True, returns fig and axs.
        """
        if not self._validate_view_prerequisites():
            return

        num_fields = len(self.fields)
        num_defocus_steps = self.num_steps

        global_axis_limit = self._compute_global_axis_limit(buffer)
        x_label, y_label = self._get_plot_axis_labels()

        total_fig_width = num_fields * figsize_per_plot[0]
        total_fig_height = num_defocus_steps * figsize_per_plot[1]

        active_theme = themes.get_active_theme_dict()
        prop_cycle = active_theme.get("axes.prop_cycle", None)
        default_theme_colors = [
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
        active_theme_colors = (
            [item["color"] for item in prop_cycle]
            if prop_cycle
            else default_theme_colors
        )

        plot_callbacks = []
        for i_pos_idx in range(num_defocus_steps):
            for j_field_idx in range(num_fields):
                callback_partial = functools.partial(
                    self._render_single_spot_diagram_subplot,
                    field_coord=self.fields[j_field_idx],
                    position_val=self.positions[i_pos_idx],
                    spot_data_list_all_wl=self.results[i_pos_idx][j_field_idx],
                    row_idx=i_pos_idx,
                    col_idx=j_field_idx,
                    num_total_rows=num_defocus_steps,
                    x_label=x_label,
                    y_label=y_label,
                    global_axis_limit=global_axis_limit,
                    theme_settings=active_theme,
                    active_theme_colors=active_theme_colors,
                )

                def final_callback(
                    ax, flat_index, p=callback_partial
                ):  # Use different name for index
                    p(ax=ax)

                plot_callbacks.append(final_callback)

        fig, axs = Plotter.plot_subplots(
            num_rows=num_defocus_steps,
            num_cols=num_fields,
            plot_callbacks=plot_callbacks,
            sharex=True,
            sharey=True,
            return_fig_ax=True,
            figsize=(total_fig_width, total_fig_height),  # Pass figsize directly
        )

        if fig:
            self._add_figure_legend_for_plotter(
                fig, active_theme_colors, figsize_per_plot, num_defocus_steps
            )
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        return Plotter.finalize_plot_objects(return_fig_ax, fig, axs)

    def _validate_view_prerequisites(self):
        """Validates prerequisites before plotting."""
        if not self.results:
            print("No data to display. Run analysis first.")
            return False
        if not self.fields or not self.wavelengths or self.num_steps == 0:
            print("No fields, defocus steps, or wavelengths to plot.")
            return False
        return True

    def _get_plot_axis_labels(self):
        """Determines axis labels based on image surface orientation."""
        cs = self.optic.image_surface.geometry.cs
        orientation = np.abs(be.to_numpy(cs.get_effective_rotation_euler()))
        tol = 0.01
        if orientation[0] > tol or orientation[1] > tol:
            return "U (mm)", "V (mm)"
        return "X (mm)", "Y (mm)"

    def _compute_global_axis_limit(self, buffer):  # Logic OK
        """Computes a global axis limit for consistent plot scaling."""
        max_r_sq = 0.0
        for (
            data_at_step
        ) in self.results:  # data_at_step is List[List[SpotData]] (field_data)
            for (
                field_spot_data_list
            ) in data_at_step:  # field_spot_data_list is List[SpotData] (wl)
                centroid_x, centroid_y = self._get_spot_centroid(field_spot_data_list)
                for spot_data_item in field_spot_data_list:
                    valid = spot_data_item.intensity != 0
                    if be.any(valid):
                        dx = spot_data_item.x - centroid_x
                        dy = spot_data_item.y - centroid_y
                        r_sq = dx[valid] ** 2 + dy[valid] ** 2
                        max_r_sq = max(max_r_sq, be.to_numpy(be.max(r_sq)).item())
        return np.sqrt(max_r_sq) * buffer if max_r_sq > 0 else 0.01

    def _get_spot_centroid(self, spot_data_list_for_field):  # Logic OK
        """Computes the centroid of spot data for the primary wavelength."""
        idx = self.optic.wavelengths.primary_index
        idx = min(idx, len(spot_data_list_for_field) - 1)
        primary_wl_spot_data = spot_data_list_for_field[idx]

        nonzero = primary_wl_spot_data.intensity != 0
        if be.any(nonzero):
            cx = be.to_numpy(be.mean(primary_wl_spot_data.x[nonzero])).item()
            cy = be.to_numpy(be.mean(primary_wl_spot_data.y[nonzero])).item()
        else:
            cx = cy = 0.0
        return cx, cy

    # _render_single_spot_diagram_subplot combines old helper methods.
    # _add_figure_legend_for_plotter is new name for _add_legend.

    def _add_figure_legend_for_plotter(
        self, fig, active_theme_colors, figsize_per_plot, num_defocus_steps
    ):
        """Adds a wavelength legend below the plot grid."""
        legend_handles = []
        legend_labels = []
        markers = ["o", "s", "^"]

        for k_wl, wl_obj in enumerate(self.wavelengths):
            color = active_theme_colors[k_wl % len(active_theme_colors)]
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=markers[k_wl % len(markers)],
                    color=color,
                    linestyle="None",
                    markersize=7,
                    alpha=0.7,
                )
            )
            legend_labels.append(
                f"{wl_obj:.4f} µm"
            )  # wl_obj is float, not Wavelength object

        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(5, len(legend_labels)),
                bbox_to_anchor=(
                    0.5,
                    -0.05 / (figsize_per_plot[1] * num_defocus_steps / 4),
                ),
            )
