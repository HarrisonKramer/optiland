"""Ray Aberration Fan Analysis

This module provides a ray fan analysis for optical systems.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt
import numpy as np


class RayFan:
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
        self.optic = optic
        self.fields = fields
        self.wavelengths = wavelengths
        if num_points % 2 == 0:
            num_points += 1  # force to be odd so a point lies at P=0
        self.num_points = num_points

        if self.fields == "all":
            self.fields = self.optic.fields.get_field_coords()

        if self.wavelengths == "all":
            self.wavelengths = self.optic.wavelengths.get_wavelengths()

        self.data = self._generate_data()

    def view(self, figsize=(10, 3.33)):
        """Displays the ray fan plot.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (10, 3.33).

        """
        _, axs = plt.subplots(
            nrows=len(self.fields),
            ncols=2,
            figsize=(figsize[0], figsize[1] * len(self.fields)),
            sharex=True,
            sharey=True,
        )

        # Ensure axs is a 2D array
        axs = np.atleast_2d(axs)

        Px = self.data["Px"]
        Py = self.data["Py"]

        for k, field in enumerate(self.fields):
            for wavelength in self.wavelengths:
                ex = self.data[f"{field}"][f"{wavelength}"]["x"]
                i_x = self.data[f"{field}"][f"{wavelength}"]["intensity_x"]
                ex[i_x == 0] = np.nan

                ey = self.data[f"{field}"][f"{wavelength}"]["y"]
                i_y = self.data[f"{field}"][f"{wavelength}"]["intensity_y"]
                ey[i_y == 0] = np.nan

                axs[k, 0].plot(Py, ey, zorder=3, label=f"{wavelength:.4f} µm")
                axs[k, 0].grid()
                axs[k, 0].axhline(y=0, lw=1, color="gray")
                axs[k, 0].axvline(x=0, lw=1, color="gray")
                axs[k, 0].set_xlabel("$P_y$")
                axs[k, 0].set_ylabel("$\\epsilon_y$ (mm)")
                axs[k, 0].set_xlim((-1, 1))
                axs[k, 0].set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

                axs[k, 1].plot(Px, ex, zorder=3, label=f"{wavelength:.4f} µm")
                axs[k, 1].grid()
                axs[k, 1].axhline(y=0, lw=1, color="gray")
                axs[k, 1].axvline(x=0, lw=1, color="gray")
                axs[k, 1].set_xlabel("$P_x$")
                axs[k, 1].set_ylabel("$\\epsilon_x$ (mm)")
                axs[k, 0].set_xlim((-1, 1))
                axs[k, 1].set_title(f"Hx: {field[0]:.3f}, Hy: {field[1]:.3f}")

        plt.legend(loc="upper center", bbox_to_anchor=(-0.1, -0.2), ncol=3)
        plt.subplots_adjust(top=1)
        plt.show()

    def _generate_data(self):
        """Generates the ray fan data.

        Returns:
            dict: The generated ray fan data.

        """
        data = {}
        data["Px"] = np.linspace(-1, 1, self.num_points)
        data["Py"] = np.linspace(-1, 1, self.num_points)
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
                    -1,
                    :,
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
                    -1,
                    :,
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
                data[f"{field}"][f"{wavelength}"]["x"] -= x_offset
                data[f"{field}"][f"{wavelength}"]["y"] -= y_offset

        return data
