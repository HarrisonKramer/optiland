"""Field Curvature Analysis

This module provides a field curvature analysis for optical systems.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt
import numpy as np


class FieldCurvature:
    """Represents a class for analyzing field curvature of an optic.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Defaults to 'all'.
        num_points (int, optional): The number of points to generate for the
            analysis. Defaults to 128.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_points (int): The number of points generated for the analysis.
        data (list): The generated data for the analysis.

    Methods:
        view(figsize=(8, 5.5)): Displays a plot of the field curvature
            analysis.

    """

    def __init__(self, optic, wavelengths="all", num_points=128):
        self.optic = optic
        if wavelengths == "all":
            wavelengths = self.optic.wavelengths.get_wavelengths()
        self.wavelengths = wavelengths
        self.num_points = num_points
        self.data = self._generate_data()

    def view(self, figsize=(8, 5.5)):
        """Displays a plot of the field curvature analysis.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (8, 5.5).

        """
        fig, ax = plt.subplots(figsize=figsize)

        field = np.linspace(0, self.optic.fields.max_field, self.num_points)

        for k, wavelength in enumerate(self.wavelengths):
            ax.plot(
                self.data[k][0],
                field,
                f"C{k}",
                zorder=10,
                label=f"{wavelength:.4f} µm, Tangential",
            )
            ax.plot(
                self.data[k][1],
                field,
                f"C{k}--",
                zorder=10,
                label=f"{wavelength:.4f} µm, Sagittal",
            )

        ax.set_xlabel("Image Plane Delta (mm)")
        ax.set_ylabel("Field")

        ax.set_ylim([0, self.optic.fields.max_field])
        current_xlim = plt.xlim()
        ax.set_xlim([-max(np.abs(current_xlim)), max(np.abs(current_xlim))])
        ax.set_title("Field Curvature")
        plt.axvline(x=0, color="k", linewidth=0.5)
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        fig.tight_layout()
        plt.show()

    def _generate_data(self):
        """Generates field curvature data for each wavelength by calculating the
            tangential and sagittal intersections.

        Returns:
            list: A list of np.ndarry containing the tangential and sagittal
                intersection points for each wavelength.

        """
        data = []
        for wavelength in self.wavelengths:
            tangential = self._intersection_parabasal_tangential(wavelength)
            sagittal = self._intersection_parabasal_sagittal(wavelength)

            data.append([tangential, sagittal])

        return data

    def _intersection_parabasal_tangential(self, wavelength, delta=1e-5):
        """Calculate the intersection of parabasal rays in tangential plane.

        Args:
            wavelength (float): The wavelength of the light.
            delta (float, optional): The delta value in normalized pupil y
                coordinates for pairs of parabasal rays. Defaults to 1e-5.

        Returns:
            numpy.ndarray: The calculated intersection values.

        """
        Hx = np.zeros(2 * self.num_points)
        Hy = np.repeat(np.linspace(0, 1, self.num_points), 2)

        Px = np.zeros(2 * self.num_points)
        Py = np.tile(np.array([-delta, delta]), self.num_points)

        self.optic.trace_generic(Hx, Hy, Px, Py, wavelength=wavelength)

        M1 = self.optic.surface_group.M[-1, ::2]
        N1 = self.optic.surface_group.N[-1, ::2]

        M2 = self.optic.surface_group.M[-1, 1::2]
        N2 = self.optic.surface_group.N[-1, 1::2]

        y01 = self.optic.surface_group.y[-1, ::2]
        z01 = self.optic.surface_group.z[-1, ::2]

        y02 = self.optic.surface_group.y[-1, 1::2]
        z02 = self.optic.surface_group.z[-1, 1::2]

        t1 = (M2 * z01 - M2 * z02 - N2 * y01 + N2 * y02) / (M1 * N2 - M2 * N1)

        return t1 * N1

    def _intersection_parabasal_sagittal(self, wavelength, delta=1e-5):
        """Calculate the intersection of parabasal rays in sagittal plane.

        Args:
            wavelength (float): The wavelength of the light.
            delta (float, optional): The delta value in normalized pupil y
                coordinates for pairs of parabasal rays. Defaults to 1e-5.

        Returns:
            numpy.ndarray: The calculated intersection values.

        """
        Hx = np.zeros(2 * self.num_points)
        Hy = np.repeat(np.linspace(0, 1, self.num_points), 2)

        Px = np.tile(np.array([-delta, delta]), self.num_points)
        Py = np.zeros(2 * self.num_points)

        self.optic.trace_generic(Hx, Hy, Px, Py, wavelength=wavelength)

        L1 = self.optic.surface_group.L[-1, ::2]
        N1 = self.optic.surface_group.N[-1, ::2]

        L2 = self.optic.surface_group.L[-1, 1::2]
        N2 = self.optic.surface_group.N[-1, 1::2]

        x01 = self.optic.surface_group.x[-1, ::2]
        z01 = self.optic.surface_group.z[-1, ::2]

        x02 = self.optic.surface_group.x[-1, 1::2]
        z02 = self.optic.surface_group.z[-1, 1::2]

        t2 = (L2 * z01 - L2 * z02 - N2 * x01 + N2 * x02) / (L1 * N2 - L2 * N1)

        return t2 * N1
