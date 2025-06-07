"""Through Focus MTF

This module provides a class for performing through-focus MTF
analysis, calculating the MTF at various focal planes for a given
spatial frequency, wavelength, and fields.

Kramer Harrison, 2025
"""

import numpy as np
from scipy.interpolate import make_interp_spline

import optiland.backend as be
from optiland.analysis.through_focus import ThroughFocusAnalysis
from optiland.mtf import SampledMTF
from optiland.plotting import LegendConfig, Plotter, config, themes  # Updated imports


class ThroughFocusMTF(ThroughFocusAnalysis):
    """
    Performs Modulation Transfer Function (MTF) analysis across a range of focal
    positions.

    This class calculates the MTF at a specified spatial frequency for both
    tangential and sagittal orientations at multiple focal planes around the
    nominal focus of an optical system.

    The results include tangential and sagittal MTF values for each analyzed
    field at each focal step.

    Args:
        optic: The optiland.optic.Optic object to analyze.
        spatial_frequency (float): The single spatial frequency (in cycles/mm)
            at which to calculate MTF. The calculation will be performed
            for both tangential (fx = spatial_frequency, fy = 0) and
            sagittal (fx = 0, fy = spatial_frequency) orientations.
        delta_focus (float, optional): The increment of focal shift in mm.
            Defaults to 0.1.
        num_steps (int, optional): The number of focal planes to analyze
            before and after the nominal focus. Must be an odd integer.
            Defaults to 5.
        fields (list[tuple[float, float]] | str, optional): Fields for
            analysis. If "all", uses all fields from `optic.fields`.
            Defaults to "all".
        wavelength (float | str, optional): The wavelength (in µm) for
            analysis. If "primary", uses the primary wavelength from
            `optic.primary_wavelength`. Defaults to "primary".
        num_rays (int, optional): The number of rays across the pupil in 1D
            for the SampledMTF calculation. Defaults to 64.
    """

    MAX_STEPS = 15
    MIN_STEPS = 1

    def __init__(
        self,
        optic,
        spatial_frequency,
        delta_focus=0.1,
        num_steps=5,
        fields="all",
        wavelength="primary",
        num_rays=128,
    ):
        self.spatial_frequency = spatial_frequency
        self.num_rays = num_rays

        if wavelength == "primary":
            self.wavelength = optic.primary_wavelength
        else:
            self.wavelength = wavelength

        super().__init__(
            optic,
            delta_focus=delta_focus,
            num_steps=num_steps,
            fields=fields,
            wavelengths=[
                self.wavelength
            ],  # Pass as list as base class expects iterable
        )

    def _perform_analysis_at_focus(self):
        """
        Performs the MTF analysis at the current focal position for all fields.
        (Code logic remains the same as original)
        """
        results_at_this_focus = []
        for field_coord in self.fields:
            sampled_mtf = SampledMTF(
                optic=self.optic,
                field=field_coord,
                wavelength=self.wavelength,  # self.wavelength is a single value
                num_rays=self.num_rays,
                distribution="uniform",
                zernike_terms=37,
                zernike_type="fringe",
            )
            freq_tan = (self.spatial_frequency, 0.0)
            freq_sag = (0.0, self.spatial_frequency)
            mtf_t = sampled_mtf.calculate_mtf([freq_tan])[0]
            mtf_s = sampled_mtf.calculate_mtf([freq_sag])[0]
            results_at_this_focus.append({"tangential": mtf_t, "sagittal": mtf_s})
        return results_at_this_focus

    def view(self, figsize=(12, 4), return_fig_ax: bool = False):
        """
        Visualizes the through-focus MTF results using Plotter.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (12, 4).
            return_fig_ax (bool, optional): If True, returns fig and ax.
                Defaults to False.
        """
        if figsize:
            original_figsize = config.get_config("figure.figsize")
            config.set_config("figure.figsize", figsize)

        fig, ax = None, None

        np_positions = be.to_numpy(be.asarray(self.positions))
        np_nominal_focus = be.to_numpy(be.asarray(self.nominal_focus))
        defocus_values_np = np_positions - np_nominal_focus

        # self.wavelengths is a list of values from BaseAnalysis.
        # ThroughFocusMTF analyzes only one wavelength.
        plot_title = (
            f"Through-Focus MTF at {self.spatial_frequency} "
            f"cycles/mm, λ={self.wavelengths[0]:.3f} µm"
        )
        xlabel = "Defocus (mm)"
        ylabel = "MTF"

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
        plot_colors = (
            [item["color"] for item in prop_cycle] if prop_cycle else default_colors
        )

        for i_field, field_coord in enumerate(self.fields):
            mtf_t_values = be.to_numpy(
                be.asarray(
                    [
                        self.results[i_pos][i_field]["tangential"]
                        for i_pos in range(self.num_steps)
                    ]
                )
            )
            mtf_s_values = be.to_numpy(
                be.asarray(
                    [
                        self.results[i_pos][i_field]["sagittal"]
                        for i_pos in range(self.num_steps)
                    ]
                )
            )
            num_data_points = len(defocus_values_np)
            Hx, Hy = field_coord
            current_field_color = plot_colors[i_field % len(plot_colors)]

            k = 3 if num_data_points >= 4 else (1 if num_data_points >= 2 else 0)

            common_plot_args = {"return_fig_ax": True, "color": current_field_color}

            if k == 0:  # Plot raw data
                label_t = f"Hx: {Hx:.1f}, Hy: {Hy:.1f}, Tangential (raw)"
                label_s = f"Hx: {Hx:.1f}, Hy: {Hy:.1f}, Sagittal (raw)"
                plot_data_t = np.clip(mtf_t_values, 0, 1)
                plot_data_s = np.clip(mtf_s_values, 0, 1)
                plot_x_data = defocus_values_np
                marker_t, marker_s = "o", "x"
                markersize = 4
            else:  # Plot smoothed data
                plot_x_data = np.linspace(
                    defocus_values_np.min(), defocus_values_np.max(), 256
                )
                spl_t = make_interp_spline(
                    defocus_values_np, mtf_t_values, k=k, check_finite=False
                )
                plot_data_t = np.clip(spl_t(plot_x_data), 0, 1)
                spl_s = make_interp_spline(
                    defocus_values_np, mtf_s_values, k=k, check_finite=False
                )
                plot_data_s = np.clip(spl_s(plot_x_data), 0, 1)
                label_t = f"Hx: {Hx:.1f}, Hy: {Hy:.1f}, Tangential"
                label_s = f"Hx: {Hx:.1f}, Hy: {Hy:.1f}, Sagittal"
                marker_t, marker_s = None, None  # No markers for smooth lines
                markersize = None

            if fig is None:
                fig, ax = Plotter.plot_line(
                    plot_x_data,
                    plot_data_t,
                    title=plot_title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend_label=label_t,
                    linestyle="-",
                    marker=marker_t,
                    markersize=markersize,
                    **common_plot_args,
                )
            else:
                Plotter.plot_line(
                    plot_x_data,
                    plot_data_t,
                    ax=ax,
                    legend_label=label_t,
                    linestyle="-",
                    marker=marker_t,
                    markersize=markersize,
                    **common_plot_args,
                )

            Plotter.plot_line(  # Sagittal always plotted on existing ax
                plot_x_data,
                plot_data_s,
                ax=ax,
                legend_label=label_s,
                linestyle="--",
                marker=marker_s,
                markersize=markersize,
                **common_plot_args,
            )

        if ax:
            ax.set_xlim([np.min(defocus_values_np), np.max(defocus_values_np)])
            ax.set_ylim([0, 1.05])
            ax.grid(
                True, linestyle=":", alpha=0.5
            )  # Customize grid as Plotter's default is simpler

            legend_cfg_params = LegendConfig(
                bbox_to_anchor=(1.05, 0.5), loc="center left", show_legend=True
            )
            if legend_cfg_params.get("show_legend", config.get_config("legend.show")):
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(
                        bbox_to_anchor=legend_cfg_params.get("bbox_to_anchor"),
                        loc=legend_cfg_params.get("loc"),
                        frameon=config.get_config("legend.frameon"),
                        shadow=config.get_config("legend.shadow"),
                        fancybox=config.get_config("legend.fancybox"),
                        ncol=config.get_config("legend.ncol"),
                        fontsize=config.get_config("font.size_legend"),
                    )

        if figsize:
            config.set_config("figure.figsize", original_figsize)

        return Plotter.finalize_plot_objects(return_fig_ax, fig, ax)
