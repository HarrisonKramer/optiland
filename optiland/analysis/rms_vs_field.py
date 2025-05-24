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

    def view(self, figsize=(7, 4.5)):
        """View the RMS spot size versus field coordinate.

        Args:
            figsize (tuple): the figure size of the output window.
                Default is (7, 4.5).

        Returns:
            None

        """
        fig, ax = plt.subplots(figsize=figsize)

        wavelengths = self.optic.wavelengths.get_wavelengths()
        labels = [f"{wavelength:.4f} µm" for wavelength in wavelengths]
        ax.plot(
            be.to_numpy(self._field[:, 1]), be.to_numpy(self._spot_size), label=labels
        )

        ax.set_xlabel("Normalized Y Field Coordinate")
        ax.set_ylabel("RMS Spot Size (mm)")

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        plt.tight_layout()
        plt.xlim(0, 1)
        plt.ylim(0, None)
        plt.grid()
        plt.tight_layout()
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

    def view(self, figsize=(7, 4.5)):
        """View the RMS wavefront error versus field coordinate.

        Args:
            figsize (tuple): the figure size of the output window.
                Default is (7, 4.5).

        Returns:
            None

        """
        fig, ax = plt.subplots(figsize=figsize)

        wavelengths = self.optic.wavelengths.get_wavelengths()
        labels = [f"{wavelength:.4f} µm" for wavelength in wavelengths]
        ax.plot(
            be.to_numpy(self._field[:, 1]),
            be.to_numpy(self._wavefront_error),
            label=labels,
        )

        ax.set_xlabel("Normalized Y Field Coordinate")
        ax.set_ylabel("RMS Wavefront Error (waves)")

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        plt.tight_layout()
        plt.xlim(0, 1)
        plt.ylim(0, None)
        plt.grid()
        plt.tight_layout()
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
