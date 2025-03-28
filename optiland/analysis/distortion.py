"""Distortion Analysis

This module provides a distortion analysis for optical systems.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt
import numpy as np


class Distortion:
    """Represents a distortion analysis for an optic.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points to generate for the
            analysis. Defaults to 128.
        distortion_type (str, optional): The type of distortion analysis.
            Defaults to 'f-tan'.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of points generated for the analysis.
        distortion_type (str): The type of distortion analysis.
        data (list): The generated distortion data.

    Methods:
        view(figsize=(7, 5.5)): Visualizes the distortion analysis.

    """

    def __init__(
        self,
        optic,
        wavelengths="all",
        num_points=128,
        distortion_type="f-tan",
    ):
        self.optic = optic
        if wavelengths == "all":
            wavelengths = self.optic.wavelengths.get_wavelengths()
        self.wavelengths = wavelengths
        self.num_points = num_points
        self.distortion_type = distortion_type
        self.data = self._generate_data()

    def view(self, figsize=(7, 5.5)):
        """Visualize the distortion analysis.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).

        """
        _, ax = plt.subplots(figsize=figsize)
        ax.axvline(x=0, color="k", linewidth=1, linestyle="--")

        field = np.linspace(1e-10, self.optic.fields.max_field, self.num_points)
        for k, wavelength in enumerate(self.wavelengths):
            ax.plot(self.data[k], field, label=f"{wavelength:.4f} µm")
            ax.set_xlabel("Distortion (%)")
            ax.set_ylabel("Field")

        current_xlim = plt.xlim()
        plt.xlim([-max(np.abs(current_xlim)), max(np.abs(current_xlim))])
        plt.ylim([0, None])
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        plt.show()

    def _generate_data(self):
        """Generate data for analysis.

        This method generates the distortion data to be used for plotting.

        Returns:
            list: A list of distortion data points.

        """
        Hx = np.zeros(self.num_points)
        Hy = np.linspace(1e-10, 1, self.num_points)

        data = []
        for wavelength in self.wavelengths:
            self.optic.trace_generic(Hx=Hx, Hy=Hy, Px=0, Py=0, wavelength=wavelength)
            yr = self.optic.surface_group.y[-1, :]

            const = yr[0] / (np.tan(1e-10 * np.radians(self.optic.fields.max_field)))

            if self.distortion_type == "f-tan":
                yp = const * np.tan(Hy * np.radians(self.optic.fields.max_field))
            elif self.distortion_type == "f-theta":
                yp = const * Hy * np.radians(self.optic.fields.max_field)
            else:
                raise ValueError(
                    '''Distortion type must be "f-tan" or
                                 "f-theta"'''
                )

            data.append(100 * (yr - yp) / yp)

        return data
