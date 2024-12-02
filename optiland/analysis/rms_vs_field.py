import numpy as np
import matplotlib.pyplot as plt
from optiland.analysis import SpotDiagram


class RmsSpotSizeVersusField(SpotDiagram):
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
    def __init__(self, optic, num_fields=64, wavelengths='all', num_rings=6,
                 distribution='hexapolar'):
        fields = [(0, Hy) for Hy in np.linspace(0, 1, num_fields)]
        super().__init__(optic, fields, wavelengths, num_rings, distribution)

        self._field = np.array(fields)
        self._spot_size = np.array(self.rms_spot_radius())

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
        labels = [f'{wavelength:.4f} µm' for wavelength in wavelengths]
        ax.plot(self._field[:, 1], self._spot_size, label=labels)

        ax.set_xlabel('Normalized Y Field Coordinate')
        ax.set_ylabel('RMS Spot Size (mm)')

        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        plt.tight_layout()
        plt.xlim(0, 1)
        plt.ylim(0, None)
        plt.grid()
        plt.tight_layout()
        plt.show()
