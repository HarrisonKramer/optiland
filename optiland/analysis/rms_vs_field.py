"""RMS versus Field Analysis

This module enables the calculation of both the RMS spot size and the RMS
wavefront error versus field coordinate of an optical system.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.analysis import SpotDiagram
from optiland.wavefront import Wavefront


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

    def view(self, fig_to_plot_on=None, figsize=(7, 4.5)):
        """View the RMS spot size versus field coordinate."""
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        analysis_wavelengths = self.wavelengths
        spot_size_data = be.to_numpy(self._spot_size)

        # Plot each wavelength's data as a separate line to handle legends correctly.
        for i, wavelength in enumerate(analysis_wavelengths):
            ax.plot(
                be.to_numpy(self._field[:, 1]),
                spot_size_data[:, i],
                label=f"{wavelength:.4f} µm",
            )

        ax.set_xlabel("Normalized Y Field Coordinate")
        ax.set_ylabel("RMS Spot Size (mm)")
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid()
        current_fig.tight_layout()

        if is_gui_embedding:
            if hasattr(current_fig, "canvas"):
                current_fig.canvas.draw_idle()
        else:
            plt.show()


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

    def view(self, fig_to_plot_on=None, figsize=(7, 4.5)):
        """View the RMS wavefront error versus field coordinate."""
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        analysis_wavelengths = self.wavelengths
        wavefront_error_data = be.to_numpy(self._wavefront_error)

        # Plot each wavelength's data as a separate line.
        for i, wavelength in enumerate(analysis_wavelengths):
            ax.plot(
                be.to_numpy(self._field[:, 1]),
                wavefront_error_data[:, i],
                label=f"{wavelength:.4f} µm",
            )
        ax.set_xlabel("Normalized Y Field Coordinate")
        ax.set_ylabel("RMS Wavefront Error (waves)")
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid()
        current_fig.tight_layout()

        if is_gui_embedding:
            if hasattr(current_fig, "canvas"):
                current_fig.canvas.draw_idle()
        else:
            plt.show()

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
