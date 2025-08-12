"""
This module defines the OPDFan class.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be

from .wavefront import Wavefront


class OPDFan(Wavefront):
    """Represents a fan plot of the wavefront error for a given optic.

    Args:
        optic (Optic): The optic for which the wavefront error is calculated.
        fields (str or list, optional): The fields for which the wavefront
            error is calculated. Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths for which the
            wavefront error is calculated. Defaults to 'all'.
        num_rays (int, optional): The number of rays used to calculate the
            wavefront error. Defaults to 100.

    Attributes:
        pupil_coord (be.ndarray): The coordinates of the pupil.
        data (list): A nested list where `data[field_idx][wavelength_idx]`
            contains `WavefrontData` for that specific field and wavelength.
            This is populated by the parent `Wavefront` class.

    Methods:
        view: Plots the wavefront error.

    """

    def __init__(self, optic, fields="all", wavelengths="all", num_rays=100, **kwargs):
        self.pupil_coord = be.linspace(-1, 1, num_rays)
        super().__init__(
            optic,
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution="cross",
            **kwargs,
        )

    def view(
        self, fig_to_plot_on: plt.Figure = None, figsize: tuple[float, float] = (10, 3)
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Visualizes the wavefront error for different fields and wavelengths.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
            figsize (tuple, optional): The size of the figure.
                Defaults to (10, 3).
        Returns:
            tuple: A tuple containing the figure and axes objects.

        Raises:
            ValueError: If the number of fields is not equal to the number of
            wavelengths, or if the number of fields is not equal to the
            number of rays.
        """
        num_rows = len(self.fields)
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            axs = current_fig.add_subplots(
                nrows=len(self.fields),
                ncols=2,
                figsize=(figsize[0], num_rows * figsize[1]),
                sharex=True,
                sharey=True,
            )
        else:
            current_fig, axs = plt.subplots(
                nrows=len(self.fields),
                ncols=2,
                figsize=(figsize[0], num_rows * figsize[1]),
                sharex=True,
                sharey=True,
            )

        # assure axes is a 2D array
        axs = np.atleast_2d(axs)

        for i, field in enumerate(self.fields):
            for wavelength in self.wavelengths:
                data = self.get_data(field, wavelength)

                wx = data.opd[self.num_rays :]
                wy = data.opd[: self.num_rays]

                intensity_x = data.intensity[self.num_rays :]
                intensity_y = data.intensity[: self.num_rays]

                wx[intensity_x == 0] = np.nan
                wy[intensity_y == 0] = np.nan

                axs[i, 0].plot(
                    be.to_numpy(self.pupil_coord),
                    be.to_numpy(wy),
                    zorder=3,
                    label=f"{wavelength:.4f} µm",
                )
                axs[i, 0].grid()
                axs[i, 0].axhline(y=0, lw=1, color="gray")
                axs[i, 0].axvline(x=0, lw=1, color="gray")
                axs[i, 0].set_xlabel("$P_y$")
                axs[i, 0].set_ylabel("Wavefront Error (waves)")
                axs[i, 0].set_xlim((-1, 1))
                axs[i, 0].set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

                axs[i, 1].plot(
                    be.to_numpy(self.pupil_coord),
                    be.to_numpy(wx),
                    zorder=3,
                    label=f"{wavelength:.4f} µm",
                )
                axs[i, 1].grid()
                axs[i, 1].axhline(y=0, lw=1, color="gray")
                axs[i, 1].axvline(x=0, lw=1, color="gray")
                axs[i, 1].set_xlabel("$P_x$")
                axs[i, 1].set_ylabel("Wavefront Error (waves)")
                axs[i, 1].set_xlim((-1, 1))
                axs[i, 1].set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

        axs[-1, -1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        current_fig.subplots_adjust(top=1)
        current_fig.tight_layout()
        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()

        return current_fig, axs
