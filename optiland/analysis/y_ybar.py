"""Y Y-bar Analysis

This module provides a y y-bar analysis for optical systems.
This is a plot of the marginal ray height versus the chief ray height
for each surface in the system.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt


class YYbar:
    """Class representing the YYbar analysis of an optic.

    Args:
        optic (Optic): The optic object to analyze.
        wavelength (str, optional): The wavelength to use for analysis.
            Defaults to 'primary'.

    Methods:
        view(figsize=(7, 5.5)): Visualizes the YYbar analysis.

    """

    def __init__(self, optic, wavelength="primary"):
        self.optic = optic
        if wavelength == "primary":
            wavelength = optic.primary_wavelength
        self.wavelength = wavelength

    def view(self, figsize=(7, 5.5)):
        """Visualizes the ray heights of the marginal and chief rays.

        Args:
            figsize (tuple): The size of the figure (width, height).
                Default is (7, 5.5).

        """
        _, ax = plt.subplots(figsize=figsize)

        ya, _ = self.optic.paraxial.marginal_ray()
        yb, _ = self.optic.paraxial.chief_ray()

        ya = ya.flatten()
        yb = yb.flatten()

        for k in range(2, len(ya)):
            label = ""
            if k == 2:
                label = "Surface 1"
            elif k == len(ya) - 1:
                label = "Image"
            ax.plot(
                [yb[k - 1], yb[k]],
                [ya[k - 1], ya[k]],
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
