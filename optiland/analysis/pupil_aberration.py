"""Pupil Aberration Analysis

The pupil abberration is defined as the difference between the paraxial
and real ray intersection point at the stop surface of the optic. This is
specified as a percentage of the on-axis paraxial stop radius at the
primary wavelength.

Kramer Harrison, 2024
"""

from matplotlib.lines import Line2D  # For custom figure legend

import optiland.backend as be
from optiland.plotting import Plotter, themes  # Updated imports

from .base import BaseAnalysis


class PupilAberration(BaseAnalysis):
    """Represents the pupil aberrations of an optic.

    The pupil abberration is defined as the difference between the paraxial
    and real ray intersection point at the stop surface of the optic. This is
    specified as a percentage of the on-axis paraxial stop radius at the
    primary wavelength.

    Args:
        optic (Optic): The optic object to analyze.
        fields (str or list, optional): The fields to analyze.
            Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points in the pupil
            aberration. Defaults to 256.

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
        """Displays the pupil aberration plot using Plotter.

        Args:
            figsize (tuple, optional): Base size for each field's row of subplots.
                Total figure height adjusts. Defaults to (10, 3.33).
            return_fig_ax (bool, optional): If True, returns fig and axs.
                Defaults to False.
        """
        plot_callbacks = []
        num_fields = len(self.fields)
        if num_fields == 0:
            print("No fields to display pupil aberration for.")
            return

        calculated_figsize = (figsize[0], figsize[1] * num_fields)

        Px_all = self.data["Px"]
        Py_all = self.data["Py"]

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

        wavelength_colors = {
            wav: colors[i % len(colors)] for i, wav in enumerate(self.wavelengths)
        }

        for field_val in self.fields:
            # Tangential callback (Py vs ey)
            def tangential_callback(ax, plot_idx_not_used, current_field=field_val):
                for wavelength in self.wavelengths:
                    ey_data = self.data[f"{current_field}"][f"{wavelength}"]["y"]
                    Py_np = be.to_numpy(Py_all)
                    ey_np = be.to_numpy(ey_data)

                    Plotter.plot_line(
                        Py_np,
                        ey_np,
                        ax=ax,
                        legend_label=f"{wavelength:.4f} µm",
                        return_fig_ax=True,  # Prevent show/close
                        color=wavelength_colors[wavelength],
                    )
                # ax.grid(True) # Plotter.plot_line handles grid via _apply_ax_styling
                ax.axhline(y=0, lw=1, color=active_theme.get("grid.color", "gray"))
                ax.axvline(x=0, lw=1, color=active_theme.get("grid.color", "gray"))
                ax.set_xlabel("$P_y$")
                ax.set_ylabel("Pupil Aberration (%)")
                ax.set_xlim((-1, 1))
                ax.set_title(f"Hx: {current_field[0]:.3f}, Hy: {current_field[1]:.3f}")

            plot_callbacks.append(tangential_callback)

            # Sagittal callback (Px vs ex)
            def sagittal_callback(ax, plot_idx_not_used, current_field=field_val):
                for wavelength in self.wavelengths:
                    ex_data = self.data[f"{current_field}"][f"{wavelength}"]["x"]
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
                # ax.grid(True) # Plotter.plot_line handles grid
                ax.axhline(y=0, lw=1, color=active_theme.get("grid.color", "gray"))
                ax.axvline(x=0, lw=1, color=active_theme.get("grid.color", "gray"))
                ax.set_xlabel("$P_x$")
                ax.set_ylabel("Pupil Aberration (%)")
                ax.set_xlim((-1, 1))
                ax.set_title(f"Hx: {current_field[0]:.3f}, Hy: {current_field[1]:.3f}")

            plot_callbacks.append(sagittal_callback)

        fig, axs = Plotter.plot_subplots(
            num_rows=num_fields,
            num_cols=2,
            plot_callbacks=plot_callbacks,
            sharex=True,
            sharey=True,
            return_fig_ax=True,
            figsize=calculated_figsize,  # Pass figsize directly
        )

        if fig is not None and axs is not None:
            legend_handles = [
                Line2D([0], [0], color=wavelength_colors[wav], lw=2)
                for wav in self.wavelengths
            ]
            legend_labels = [f"{wav:.4f} µm" for wav in self.wavelengths]

            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(-0.1, -0.2),  # May need adjustment
                ncol=3,
            )
            # Adjust layout to prevent legend overlap, similar to RayFan
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # rect=[left, bottom, right, top]

        return Plotter.finalize_plot_objects(return_fig_ax, fig, axs)

    def _generate_data(self):
        """Generate the real pupil aberration data.

        Returns:
            dict: The pupil aberration data.

        """
        stop_idx = self.optic.surface_group.stop_index

        data = {
            "Px": be.linspace(-1, 1, self.num_points),
            "Py": be.linspace(-1, 1, self.num_points),
        }

        # determine size of stop
        self.optic.paraxial.trace(0, 1, self.optic.primary_wavelength)
        d = self.optic.surface_group.y[stop_idx, 0]

        # Paraxial trace
        self.optic.paraxial.trace(0, data["Py"], self.optic.primary_wavelength)
        parax_ref = self.optic.surface_group.y[stop_idx, :]

        for field in self.fields:
            Hx = field[0]
            Hy = field[1]

            data[f"{field}"] = {}
            for wavelength in self.wavelengths:
                data[f"{field}"][f"{wavelength}"] = {}

                # Trace along the x-axis
                self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_x",
                )
                real_x = self.optic.surface_group.x[stop_idx, :]
                real_int_x = self.optic.surface_group.intensity[stop_idx, :]

                # Trace along the y-axis
                self.optic.trace(
                    Hx=Hx,
                    Hy=Hy,
                    wavelength=wavelength,
                    num_rays=self.num_points,
                    distribution="line_y",
                )
                real_y = self.optic.surface_group.y[stop_idx, :]
                real_int_y = self.optic.surface_group.intensity[stop_idx, :]

                # Compute error
                error_x = (parax_ref - real_x) / d * 100
                error_x[real_int_x == 0] = be.nan

                error_y = (parax_ref - real_y) / d * 100
                error_y[real_int_y == 0] = be.nan

                data[f"{field}"][f"{wavelength}"]["x"] = error_x
                data[f"{field}"][f"{wavelength}"]["y"] = error_y

        return data
