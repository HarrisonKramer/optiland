"""Ray Aberration Fan Analysis

This module provides a ray fan analysis for optical systems.

Kramer Harrison, 2024
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be

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

    def view(
        self,
        fig_to_plot_on: plt.Figure = None,
        figsize: tuple[float, float] = (10, 3.33),
    ):
        """
        Displays the ray fan plot, either in a new window or on a provided GUI figure.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (10, 3.33).
        Returns:
            tuple: The current figure and its axes.
        """
        is_gui_embedding = fig_to_plot_on is not None
        num_fields = len(self.fields)

        if num_fields == 0:
            if is_gui_embedding:
                fig_to_plot_on.text(
                    0.5, 0.5, "No fields to plot.", ha="center", va="center"
                )
                if hasattr(fig_to_plot_on, "canvas"):
                    fig_to_plot_on.canvas.draw_idle()
            else:
                print("Warning (RayFan.view): No fields to plot.")
            return

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            axs = current_fig.subplots(
                nrows=num_fields, ncols=2, sharex=True, sharey=True
            )
        else:
            current_fig, axs = plt.subplots(
                nrows=num_fields,
                ncols=2,
                figsize=(figsize[0], figsize[1] * num_fields),
                sharex=True,
                sharey=True,
            )

        axs = np.atleast_2d(axs)
        Px, Py = self.data["Px"], self.data["Py"]

        for k, field in enumerate(self.fields):
            ax_y, ax_x = axs[k, 0], axs[k, 1]
            for wavelength in self.wavelengths:
                ex = self.data[f"{field}"][f"{wavelength}"]["x"]
                ey = self.data[f"{field}"][f"{wavelength}"]["y"]
                i_x = self.data[f"{field}"][f"{wavelength}"]["intensity_x"]
                i_y = self.data[f"{field}"][f"{wavelength}"]["intensity_y"]
                ex[i_x == 0], ey[i_y == 0] = be.nan, be.nan

                ax_y.plot(
                    be.to_numpy(Py),
                    be.to_numpy(ey),
                    zorder=3,
                    label=f"{wavelength:.4f} µm",
                )
                ax_x.plot(
                    be.to_numpy(Px),
                    be.to_numpy(ex),
                    zorder=3,
                    label=f"{wavelength:.4f} µm",
                )

            ax_y.grid()
            ax_y.axhline(0, lw=1, c="gray")
            ax_y.axvline(0, lw=1, c="gray")
            ax_y.set_xlabel("$P_y$")
            ax_y.set_ylabel("$\\epsilon_y$ (mm)")
            ax_y.set_xlim(-1, 1)
            ax_y.set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

            ax_x.grid()
            ax_x.axhline(0, lw=1, c="gray")
            ax_x.axvline(0, lw=1, c="gray")
            ax_x.set_xlabel("$P_x$")
            ax_x.set_ylabel("$\\epsilon_x$ (mm)")
            ax_x.set_xlim(-1, 1)
            ax_x.set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

        if num_fields > 0:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            if handles:
                current_fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.05 / num_fields),
                    ncol=len(self.wavelengths),
                )

        current_fig.tight_layout()
        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        return current_fig, current_fig.get_axes()

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
