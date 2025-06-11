"""Angle vs. Height Plot Analysis

This module provides classes for analyzing the incident angle versus image height
for optical systems, across both pupil and field.

BuergiR & Kramer Harrison, 2025
"""

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland import distribution

from .base import BaseAnalysis


class PupilFieldAngleVsHeight(BaseAnalysis):
    """Represents an analysis of incident angle vs. image height by varying
    through all pupil coordinates (Px, Py) for a given image field point.

    This analysis is useful for testing the telecentricity of a lens after
    a point light source (object).

    Args:
        optic (Optic): The optic object to analyze.
        surface_idx (int, optional): Index of the surface at which the angle and
            height are measured. Defaults to -1 (last surface).
        axis (int, optional): Specifies the axis for measurement. 0 for x-axis,
            1 for y-axis. Defaults to 1 (y-axis).
        wavelengths (str or list, optional): A single wavelength or a list of
            wavelengths. Passed directly to BaseAnalysis. Defaults to 'all'.
        field_point (tuple, optional): A single relative image field point (Hx, Hy).
            Defaults to (0, 0).
        num_points (int, optional): The number of points used for the plot.
            Defaults to 51.

    Attributes:
        optic (Optic): The optic object being analyzed.
        surface_idx (int): Index of the surface for measurements.
        axis (int): Axis for measurement (0 for x, 1 for y).
        wavelengths (list): The wavelengths being analyzed (handled by BaseAnalysis).
        field_point (tuple): The relative image field point.
        num_points (int): The number of points generated for the analysis.
        data (dict): The generated data for the analysis, structured as:
            {
                (Hx, Hy, wavelength_value): {'height': np.ndarray, 'angle': np.ndarray}
            }
    """

    def __init__(
        self,
        optic,
        surface_idx=-1,
        axis=1,
        wavelengths="all",
        field_point=(0, 0),
        num_points=51,
    ):
        self.surface_idx = surface_idx
        self.axis = axis
        self.field_point = field_point
        self.num_points = num_points

        # Wavelengths handling is delegated to the BaseAnalysis __init__
        super().__init__(optic, wavelengths=wavelengths)

    def _generate_data(self):
        """Generates the incident angle vs. image height data by varying through
        pupil coordinates.

        Returns:
            dict: A dictionary containing the generated data.
                Keys are (Hx, Hy, wavelength_value) tuples, values are dictionaries
                with 'height' and 'angle' numpy arrays.
        """
        data = {}
        Hx, Hy = self.field_point

        if self.axis == 1:  # Y-axis variation (Py)
            distrib = distribution.LineYDistribution()
        else:  # X-axis variation (Px)
            distrib = distribution.LineXDistribution()
        distrib.generate_points(num_points=self.num_points)

        for wavelength in self.wavelengths:
            wavelength_value = (
                wavelength.value if hasattr(wavelength, "value") else wavelength
            )

            self.optic.trace(
                Hx=Hx, Hy=Hy, wavelength=wavelength_value, distribution=distrib
            )

            if self.axis == 1:
                M = self.optic.surface_group.M[self.surface_idx, :]
                y_coords = self.optic.surface_group.y[self.surface_idx, :]
                idx = np.argsort(y_coords)
                height = y_coords[idx]
                angle = np.arcsin(M[idx])
            else:
                L = self.optic.surface_group.L[self.surface_idx, :]
                x_coords = self.optic.surface_group.x[self.surface_idx, :]
                idx = np.argsort(x_coords)
                height = x_coords[idx]
                angle = np.arcsin(L[idx])

            data[(Hx, Hy, wavelength_value)] = {
                "height": be.to_numpy(height),
                "angle": be.to_numpy(angle),
            }
        return data

    def view(self, figsize=(8, 5.5), title=None):
        """Displays a plot of the incident angle vs. image height analysis from
        the pupil field perspective.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (8, 5.5).
            title (str, optional): An optional subtitle to be added to the plot.
                If None, lens name is used.
        """
        plot_style = ".-" if self.num_points < 75 else "-"
        plot_data_list = []
        for (Hx, Hy, wavelength), plot_data in self.data.items():
            label_str = (
                f"Hx={round(Hx, 4)} Hy={round(Hy, 4)}, {np.round(wavelength, 4)} μm"
            )
            plot_data_list.append(
                (plot_data["height"], np.rad2deg(plot_data["angle"]), label_str)
            )

        _plot_angle_vs_height(
            plot_data_list=plot_data_list,
            axis=self.axis,
            optic_name=self.optic.name,
            plot_style=plot_style,
            figsize=figsize,
            title=title,
        )


class ImageFieldAngleVsHeight(BaseAnalysis):
    """Represents an analysis of incident angle vs. image height by varying
    through all image field coordinates (Hx, Hy) for a given pupil field point.

    This analysis is useful for testing the telecentricity of a scan lens with
    a scan mirror at the entrance pupil. Note: Uses trace_generic(), which is
    slower than trace().

    Args:
        optic (Optic): The optic object to analyze.
        surface_idx (int, optional): Index of the surface at which the angle and
            height are measured. Defaults to -1 (last surface).
        axis (int, optional): Specifies the axis for measurement. 0 for x-axis,
            1 for y-axis. Defaults to 1 (y-axis).
        wavelengths (str or list, optional): A single wavelength or a list of
            wavelengths. Passed directly to BaseAnalysis. Defaults to 'all'.
        pupil_point (tuple, optional): A single pupil field point (Px, Py).
            Defaults to (0, 0).
        num_points (int, optional): The number of points displayed on the plot.
            Defaults to 51.

    Attributes:
        optic (Optic): The optic object being analyzed.
        surface_idx (int): Index of the surface for measurements.
        axis (int): Axis for measurement (0 for x, 1 for y).
        wavelengths (list): The wavelengths being analyzed (handled by BaseAnalysis).
        pupil_point (tuple): The pupil field point.
        num_points (int): The number of points generated for the analysis.
        data (dict): The generated data for the analysis, structured as:
            {
                (Px, Py, wavelength_value): {'height': np.ndarray, 'angle': np.ndarray}
            }
    """

    def __init__(
        self,
        optic,
        surface_idx=-1,
        axis=1,
        wavelengths="all",
        pupil_point=(0, 0),
        num_points=51,
    ):
        self.surface_idx = surface_idx
        self.axis = axis
        self.pupil_point = pupil_point
        self.num_points = num_points

        # Wavelengths handling is delegated to the BaseAnalysis __init__
        super().__init__(optic, wavelengths=wavelengths)

    def _generate_data(self):
        """Generates the incident angle vs. image height data by varying through
        image field coordinates.

        Returns:
            dict: A dictionary containing the generated data.
                Keys are (Px, Py, wavelength_value) tuples, values are dictionaries
                with 'height' and 'angle' numpy arrays.
        """
        data = {}
        px, py = self.pupil_point

        field_range = be.linspace(start=-1, stop=1, num=self.num_points).astype(float)
        angle = be.zeros_like(field_range)
        height = be.zeros_like(field_range)

        for wavelength in self.wavelengths:
            wavelength_value = (
                wavelength.value if hasattr(wavelength, "value") else wavelength
            )

            for idx, field_val in enumerate(field_range):
                if self.axis == 1:  # Y-direction variation (Hy)
                    ray = self.optic.trace_generic(
                        Hx=0, Hy=field_val, Px=px, Py=py, wavelength=wavelength_value
                    )
                    M = ray.M[self.surface_idx]
                    angle[idx] = be.arcsin(M)
                    height[idx] = ray.y[self.surface_idx]
                elif self.axis == 0:  # X-direction variation (Hx)
                    ray = self.optic.trace_generic(
                        Hx=field_val, Hy=0, Px=px, Py=py, wavelength=wavelength_value
                    )
                    L = ray.L[self.surface_idx]
                    angle[idx] = be.arcsin(L)
                    height[idx] = ray.x[self.surface_idx]

            data[(px, py, wavelength_value)] = {
                "height": be.to_numpy(height),
                "angle": be.to_numpy(angle),
            }
        return data

    def view(self, figsize=(8, 5.5), title=None):
        """Displays a plot of the incident angle vs. image height analysis from
        the image field perspective.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (8, 5.5).
            title (str, optional): An optional subtitle to be added to the plot.
                If None, lens name is used.
        """
        plot_style = ".-" if self.num_points < 75 else "-"
        plot_data_list = []
        for (px, py, wavelength), plot_data in self.data.items():
            label_str = (
                f"Px={round(px, 4)} Py={round(py, 4)}, {np.round(wavelength, 4)} μm"
            )
            plot_data_list.append(
                (plot_data["height"], np.rad2deg(plot_data["angle"]), label_str)
            )

        _plot_angle_vs_height(
            plot_data_list=plot_data_list,
            axis=self.axis,
            optic_name=self.optic.name,
            plot_style=plot_style,
            figsize=figsize,
            title=title,
        )


def _plot_angle_vs_height(plot_data_list, axis, optic_name, plot_style, figsize, title):
    """Helper function to generate a consistent angle vs. image height plot.

    Args:
        plot_data_list (list): A list of tuples, where each tuple contains
            (height_array, angle_degrees_array, label_string).
        axis (int): Specifies the axis for measurement (0 for x, 1 for y).
        optic_name (str): The name of the optic, used for the plot title.
        plot_style (str): Matplotlib plot style (e.g., '.-', '-').
        figsize (tuple): The size of the figure.
        title (str or None): An optional subtitle for the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for height, angle_deg, label_str in plot_data_list:
        ax.plot(height, angle_deg, plot_style, label=label_str)

    fig.suptitle("Incident Angle vs Image Height" + (" (x-axis)" if axis == 0 else ""))
    ax.set_title(title if title else optic_name, fontsize=10)
    ax.set_xlabel("Image Height in Millimeters")
    ax.set_ylabel("Incident Angle in Degrees")
    ax.minorticks_on()
    ax.grid(visible=True, which="major", color="darkgrey", linestyle="-")
    ax.grid(visible=True, which="minor", color="lightgrey", linestyle="--")
    ax.legend()
    fig.tight_layout()
    plt.show()
