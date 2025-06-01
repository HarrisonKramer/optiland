"""Y Y-bar Analysis

This module provides a y y-bar analysis for optical systems.
This is a plot of the marginal ray height versus the chief ray height
for each surface in the system.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt

import optiland.backend as be

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
            super_wavelengths_arg = "primary"
        elif isinstance(wavelength, (float, int, be.Tensor)):
            if hasattr(wavelength, "item"):  # scalar tensor
                super_wavelengths_arg = [float(wavelength.item())]
            else:
                super_wavelengths_arg = [float(wavelength)]
        elif hasattr(wavelength, "value"):  # Wavelength object
            super_wavelengths_arg = [float(wavelength.value)]
        else:
            raise TypeError(
                f"Unsupported wavelength type for YYbar: {type(wavelength)}"
            )

        super().__init__(optic, wavelengths=super_wavelengths_arg)
        # self.optic and self.wavelengths are set by BaseAnalysis.
        # self.data will be set by super()'s call to self._generate_data().

    def _generate_data(self):
        # Use the first (and only expected) wavelength from the list.
        current_wavelength = self.wavelengths[0]

        # Assuming paraxial.marginal_ray and chief_ray can take a wavelength argument.
        ya, _ = self.optic.paraxial.marginal_ray(wavelength_value=current_wavelength)
        yb, _ = self.optic.paraxial.chief_ray(wavelength_value=current_wavelength)

        return {"ya": ya.flatten(), "yb": yb.flatten()}

    def view(self, figsize=(7, 5.5)):
        """Visualizes the ray heights of the marginal and chief rays.

        Args:
            figsize (tuple): The size of the figure (width, height).
                Default is (7, 5.5).

        """
        _, ax = plt.subplots(figsize=figsize)

        ya = self.data["ya"]
        yb = self.data["yb"]

        for k in range(2, len(ya)):
            label = ""
            if k == 2:
                label = "Surface 1"
            elif k == len(ya) - 1:
                label = "Image"
            ax.plot(
                [be.to_numpy(yb[k - 1]), be.to_numpy(yb[k])],
                [be.to_numpy(ya[k - 1]), be.to_numpy(ya[k])],
                ".-",
                label=label,
                markersize=8,
            )

        ax.axhline(y=0, linewidth=0.5, color="k")
        ax.axvline(x=0, linewidth=0.5, color="k")
        ax.set_xlabel("Chief Ray Height (mm)")
        ax.set_ylabel("Marginal Ray Height (mm)")
        ax.legend()
        plt.show()
