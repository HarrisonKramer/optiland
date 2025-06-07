"""RMS versus Field Analysis

This module enables the calculation of both the RMS spot size and the RMS
wavefront error versus field coordinate of an optical system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.analysis import (
    SpotDiagram,
)  # SpotDiagram might also need Plotter updates if not done
from optiland.plotting import LegendConfig, Plotter, config  # Updated imports
from optiland.wavefront import Wavefront  # Wavefront might also need Plotter updates


class RmsSpotSizeVsField(SpotDiagram):
    """RMS Spot Size versus Field Coordinate.

    This class is used to analyze the RMS spot size versus field coordinate
    of an optical system.

    Args:
        optic (Optic): the optical system.
        num_fields (int): the number of fields. Default is 64.
        wavelengths (list): the wavelengths to be analyzed. Default is 'all'.
        num_rings (int): the number of rings. Default is 6.
        distribution (str): the distribution of the fields.
            Default is 'hexapolar'.

    """

    def __init__(
        self,
        optic,
        num_fields=64,
        wavelengths="all",
        num_rings=6,
        distribution="hexapolar",
    ):
        self.num_fields = num_fields
        fields = [(0, Hy) for Hy in be.linspace(0, 1, num_fields)]
        super().__init__(optic, fields, wavelengths, num_rings, distribution)

        self._field = be.array(fields)
        self._spot_size = be.array(self.rms_spot_radius())

    def view(self, figsize=(7, 4.5), return_fig_ax: bool = False):
        """View the RMS spot size versus field coordinate using Plotter.

        Args:
            figsize (tuple): The figure size of the output window.
                Defaults to (7, 4.5).
            return_fig_ax (bool, optional): If True, returns the figure and axes
                objects. Defaults to False, which shows the plot.
        """
        if figsize:
            original_figsize = config.get_config("figure.figsize")
            config.set_config("figure.figsize", figsize)

        fig, ax = None, None
        fields_np = be.to_numpy(self._field[:, 1])

        xlabel = "Normalized Y Field Coordinate"
        ylabel = "RMS Spot Size (mm)"
        plot_title = "RMS Spot Size vs Field"  # Added a title

        # Iterate over self.wavelengths (list of values from BaseAnalysis)
        for idx, wavelength_value in enumerate(self.wavelengths):
            # Assuming self._spot_size is (num_fields, num_wavelengths)
            spot_size_np_wl = be.to_numpy(self._spot_size[:, idx])
            legend_label = f"{wavelength_value:.4f} µm"  # Use the value directly

            if fig is None:
                fig, ax = Plotter.plot_line(
                    fields_np,
                    spot_size_np_wl,
                    title=plot_title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend_label=legend_label,
                    return_fig_ax=True,
                )
            else:
                Plotter.plot_line(
                    fields_np,
                    spot_size_np_wl,
                    ax=ax,
                    legend_label=legend_label,
                    return_fig_ax=True,
                )

        if ax:
            ax.set_xlim([0, 1])
            ax.set_ylim([0, None])  # y-axis starts from 0

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
            # Plotter's _apply_ax_styling handles grid.
            # _handle_fig_ax_return_logic manages final layout.

        if figsize:
            config.set_config("figure.figsize", original_figsize)

        return Plotter.finalize_plot_objects(return_fig_ax, fig, ax)


class RmsWavefrontErrorVsField(Wavefront):
    """RMS Wavefront Error versus Field Coordinate.

    This class is used to analyze the RMS wavefront error versus field
    coordinate of an optical system.

    Args:
        optic (Optic): the optical system.
        num_fields (int): the number of fields. Default is 32.
        wavelengths (list): the wavelengths to be analyzed. Default is 'all'.
        num_rays (int): the number of rays. Default is 12.
        distribution (str): the distribution of the fields.
            Default is 'hexapolar'.

    """

    def __init__(
        self,
        optic,
        num_fields=32,
        wavelengths="all",
        num_rays=12,
        distribution="hexapolar",
    ):
        self.num_fields = num_fields
        fields = [(0, Hy) for Hy in be.linspace(0, 1, num_fields)]
        super().__init__(optic, fields, wavelengths, num_rays, distribution)

        self._field = be.array(fields)
        self._wavefront_error = be.array(self._rms_wavefront_error())

    def view(self, figsize=(7, 4.5), return_fig_ax: bool = False):
        """View the RMS wavefront error versus field coordinate using Plotter.

        Args:
            figsize (tuple): The figure size of the output window.
                Defaults to (7, 4.5).
            return_fig_ax (bool, optional): If True, returns the figure and axes
                objects. Defaults to False, which shows the plot.
        """
        if figsize:
            original_figsize = config.get_config("figure.figsize")
            config.set_config("figure.figsize", figsize)

        fig, ax = None, None
        fields_np = be.to_numpy(self._field[:, 1])

        xlabel = "Normalized Y Field Coordinate"
        ylabel = "RMS Wavefront Error (waves)"
        plot_title = "RMS Wavefront Error vs Field"  # Added a title

        # Iterate over self.wavelengths (list of values from BaseAnalysis)
        for idx, wavelength_value in enumerate(self.wavelengths):
            # Assuming self._wavefront_error is (num_fields, num_wavelengths)
            wfe_np_wl = be.to_numpy(self._wavefront_error[:, idx])
            legend_label = f"{wavelength_value:.4f} µm"  # Use the value directly

            if fig is None:
                fig, ax = Plotter.plot_line(
                    fields_np,
                    wfe_np_wl,
                    title=plot_title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend_label=legend_label,
                    return_fig_ax=True,
                )
            else:
                Plotter.plot_line(
                    fields_np,
                    wfe_np_wl,
                    ax=ax,
                    legend_label=legend_label,
                    return_fig_ax=True,
                )

        if ax:
            ax.set_xlim([0, 1])
            ax.set_ylim([0, None])  # y-axis starts from 0

            legend_cfg_params = LegendConfig(
                bbox_to_anchor=(1.05, 0.5), loc="center left", show_legend=True
            )
            if legend_cfg_params.get("show_legend", config.get_config("legend.show")):
                handles, _ = ax.get_legend_handles_labels()
                if handles:  # Ensure there's something to make a legend for
                    ax.legend(
                        bbox_to_anchor=legend_cfg_params.get("bbox_to_anchor"),
                        loc=legend_cfg_params.get("loc"),
                        frameon=config.get_config("legend.frameon"),
                        shadow=config.get_config("legend.shadow"),
                        fancybox=config.get_config("legend.fancybox"),
                        ncol=config.get_config("legend.ncol"),
                        fontsize=config.get_config("font.size_legend"),
                    )
            # Plotter's _apply_ax_styling handles grid.

        if figsize:
            config.set_config("figure.figsize", original_figsize)

        return Plotter.finalize_plot_objects(return_fig_ax, fig, ax)

    def _rms_wavefront_error(self):
        """Calculate the RMS wavefront error."""
        rows = []
        for field in self.fields:
            cols = []
            for wl in self.wavelengths:
                wavefront_data = self.get_data(field, wl)
                rms_ij = be.sqrt(be.mean(wavefront_data.opd**2))
                cols.append(rms_ij)
            # turn this row into a backend array/tensor
            rows.append(be.stack(cols, axis=0))
        # stack all rows into the final 2D result
        return be.stack(rows, axis=0)
