"""Optical System Visualization Module

This module provides tools for visualizing optical systems.
It utilizes Matplotlib to render optical components and ray tracing paths.
The `OpticViewer` class is the primary interface for generating these visualizations,
offering customization for ray properties, field of view, and display parameters.

Kramer Harrison, 2024

re-worked by Manuel Fragata Mendes, june 2025
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from optiland.visualization.base import BaseViewer
from optiland.visualization.system.rays import Rays2D
from optiland.visualization.system.system import OpticalSystem


class OpticViewer(BaseViewer):
    """A class used to visualize optical systems.

    Args:
        optic: The optical system to be visualized.

    Attributes:
        optic: The optical system to be visualized.
        rays: An instance of Rays2D for ray tracing.
        system: An instance of OpticalSystem for system representation.

    Methods:
        view(fields='all', wavelengths='primary', num_rays=3,
             distribution='line_y', figsize=(10, 4), xlim=None, ylim=None):
            Visualizes the optical system with specified parameters.

    """

    def __init__(self, optic):
        self.optic = optic

        self.rays = Rays2D(optic)
        self.system = OpticalSystem(optic, self.rays, projection="2d")

    def view(
        self,
        fields="all",
        wavelengths="primary",
        num_rays=3,
        distribution="line_y",
        figsize=(10, 4),
        xlim=None,
        ylim=None,
        title=None,
        reference=None,
    ):
        """Visualizes the optical system.

        Args:
            fields (str, optional): The fields to be visualized.
                Defaults to 'all'.
            wavelengths (str, optional): The wavelengths to be visualized.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be visualized.
                Defaults to 3.
            distribution (str, optional): The distribution of rays.
                Defaults to 'line_y'.
            figsize (tuple, optional): The size of the figure.
                Defaults to (10, 4).
            xlim (tuple, optional): The x-axis limits. Defaults to None.
            ylim (tuple, optional): The y-axis limits. Defaults to None.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.

        """
        fig, ax = plt.subplots(figsize=figsize)

        self.rays.plot(
            ax,
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution=distribution,
            reference=reference,
        )

        self.system.plot(ax)

        ax.set_facecolor("#f8f9fa")  # off-white background
        ax.axis("image")
        ax.set_xlabel("Z [mm]")
        ax.set_ylabel("Y [mm]")

        if title:
            ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(alpha=0.25)

        # Return the figure and axes instead of showing the plot
        return fig, ax
