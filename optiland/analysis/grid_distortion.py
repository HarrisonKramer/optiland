"""Grid Distortion Analysis

This module provides a grid distortion analysis for optical systems.
This is module enables calculation of the distortion over a grid of points
for an optical system.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be

from .base import BaseAnalysis


class GridDistortion(BaseAnalysis):
    """Represents a grid distortion analysis for an optical system.

    Args:
        optic (Optic): The optical system to analyze.
        wavelength (str, optional): The wavelength of light to use for
            analysis. Defaults to 'primary'.
        num_points (int, optional): The number of points along each axis of the
            grid. Defaults to 10.
        distortion_type (str, optional): The type of distortion to analyze.
            Must be 'f-tan' or 'f-theta'. Defaults to 'f-tan'.

    Attributes:
        optic (Optic): The optical system being analyzed.
        wavelength (str): The wavelength of light used for analysis.
        num_points (int): The number of points in the grid.
        distortion_type (str): The type of distortion being analyzed.
        data (dict): The generated data for the analysis.

    Methods:
        view(figsize=(7, 5.5)): Visualizes the grid distortion analysis.

    """

    def __init__(
        self,
        optic,
        wavelength="primary",
        num_points=10,
        distortion_type="f-tan",
    ):
        if isinstance(wavelength, str) and wavelength == "primary":
            processed_wavelength = "primary"
        elif isinstance(wavelength, (float, int)):
            processed_wavelength = float(wavelength)
        else:
            raise TypeError(
                f"Unsupported wavelength type: {type(wavelength)} for GridDistortion. "
                f"Expected 'primary', float, or int"
            )

        self.num_points = num_points
        self.distortion_type = distortion_type
        super().__init__(optic, wavelengths=processed_wavelength)

    def view(self, figsize=(7, 5.5)):
        """Visualizes the grid distortion analysis.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (7, 5.5).

        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            be.to_numpy(self.data["xp"]),
            be.to_numpy(self.data["yp"]),
            "C1",
            linewidth=1,
        )
        ax.plot(
            be.to_numpy(self.data["xp"]).T,
            be.to_numpy(self.data["yp"]).T,
            "C1",
            linewidth=1,
        )

        ax.plot(be.to_numpy(self.data["xr"]), be.to_numpy(self.data["yr"]), "C0P")
        ax.plot(be.to_numpy(self.data["xr"]).T, be.to_numpy(self.data["yr"]).T, "C0P")

        ax.set_xlabel("Image X (mm)")
        ax.set_ylabel("Image Y (mm)")
        ax.set_aspect("equal", adjustable="box")

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        max_distortion = self.data["max_distortion"]
        ax.set_title(f"Max Distortion: {max_distortion:.2f}%")
        fig.tight_layout()
        plt.show()

    def _generate_data(self):
        """Generates the data for the grid distortion analysis.

        Returns:
            dict: The generated data.

        Raises:
            ValueError: If the distortion type is not 'f-tan' or 'f-theta'.

        """
        # Trace chief ray to retrieve central (x, y) position
        current_wavelength = self.wavelengths[0]
        self.optic.trace_generic(
            Hx=0,
            Hy=0,
            Px=0,
            Py=0,
            wavelength=current_wavelength,
        )
        x_chief = self.optic.surface_group.x[-1, 0]
        y_chief = self.optic.surface_group.y[-1, 0]

        # Trace single reference ray
        current_wavelength = self.wavelengths[0]
        self.optic.trace_generic(
            Hx=0,
            Hy=1e-10,  # small field
            Px=0,
            Py=0,
            wavelength=current_wavelength,
        )
        y_ref = self.optic.surface_group.y[-1, 0]

        max_field = np.sqrt(2) / 2
        extent = be.linspace(-max_field, max_field, self.num_points)
        Hx, Hy = be.meshgrid(extent, extent)

        if self.distortion_type == "f-tan":
            const = (y_ref - y_chief) / (
                be.tan(1e-10 * be.radians(self.optic.fields.max_field))
            )
            xp = const * be.tan(Hx * be.radians(self.optic.fields.max_field))
            yp = const * be.tan(Hy * be.radians(self.optic.fields.max_field))
        elif self.distortion_type == "f-theta":
            const = (y_ref - y_chief) / (
                1e-10 * be.radians(self.optic.fields.max_field)
            )
            xp = const * Hx * be.radians(self.optic.fields.max_field)
            yp = const * Hy * be.radians(self.optic.fields.max_field)
        else:
            raise ValueError('''Distortion type must be "f-tan" or"f-theta"''')

        self.optic.trace_generic(
            Hx=Hx.flatten(),
            Hy=Hy.flatten(),
            Px=0,
            Py=0,
            wavelength=current_wavelength,
        )

        data = {}

        # make real grid square for ease of plotting
        data["xr"] = (
            be.reshape(
                self.optic.surface_group.x[-1, :],
                (self.num_points, self.num_points),
            )
            - x_chief
        )
        data["yr"] = (
            be.reshape(
                self.optic.surface_group.y[-1, :],
                (self.num_points, self.num_points),
            )
            - y_chief
        )

        data["xp"] = xp
        data["yp"] = yp

        # Find max distortion
        delta = be.sqrt((data["xp"] - data["xr"]) ** 2 + (data["yp"] - data["yr"]) ** 2)
        rp = be.sqrt(data["xp"] ** 2 + data["yp"] ** 2)

        data["max_distortion"] = be.max(100 * delta / rp)

        return data
