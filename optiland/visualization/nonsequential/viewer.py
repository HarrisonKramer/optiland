"""Non-sequential Scene Visualization Module (2D).

Kramer Harrison, 2026
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from optiland.visualization.base import BaseViewer
from optiland.visualization.themes import get_active_theme
from optiland.visualization.nonsequential.surface import NSQSurface2D
from optiland.visualization.nonsequential.rays import NSQRays2D


class NSQViewer(BaseViewer):
    """A class used to visualize non-sequential scenes.

    Args:
        scene: The non-sequential scene to be visualized.
    """

    def __init__(self, scene):
        self.scene = scene
        self.rays_viewer = NSQRays2D(scene)

    def view(
        self,
        figsize=None,
        xlim=None,
        ylim=None,
        title=None,
        projection="YZ",
        ax=None,
    ):
        """Visualizes the non-sequential scene.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to None, which uses the theme's default.
            xlim (tuple, optional): The x-axis limits. Defaults to None.
            ylim (tuple, optional): The y-axis limits. Defaults to None.
            title (str, optional): The title of the plot. Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.
            ax (matplotlib.axes.Axes, optional): The axes to plot on.
                If None, a new figure and axes are created. Defaults to None.
        """
        if projection not in ["XY", "XZ", "YZ"]:
            raise ValueError("Invalid projection type. Must be 'XY', 'XZ', or 'YZ'.")

        theme = get_active_theme()
        params = theme.parameters
        if figsize is None:
            figsize = params["figure.figsize"]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_facecolor(params["figure.facecolor"])
        else:
            fig = ax.get_figure()

        ax.set_facecolor(params["axes.facecolor"])

        # Plot surfaces
        for surf in self.scene.surfaces:
            surf_viewer = NSQSurface2D(surf)
            surf_viewer.plot(ax, theme=theme, projection=projection)

        # Plot rays
        self.rays_viewer.plot(ax, theme=theme, projection=projection)

        ax.axis("image")
        if projection == "YZ":
            ax.set_xlabel("Z [mm]", color=params["axes.labelcolor"])
            ax.set_ylabel("Y [mm]", color=params["axes.labelcolor"])
        elif projection == "XZ":
            ax.set_xlabel("Z [mm]", color=params["axes.labelcolor"])
            ax.set_ylabel("X [mm]", color=params["axes.labelcolor"])
        else:  # XY
            ax.set_xlabel("X [mm]", color=params["axes.labelcolor"])
            ax.set_ylabel("Y [mm]", color=params["axes.labelcolor"])

        ax.tick_params(axis="x", colors=params["xtick.color"])
        ax.tick_params(axis="y", colors=params["ytick.color"])
        ax.spines["bottom"].set_color(params["axes.edgecolor"])
        ax.spines["top"].set_color(params["axes.edgecolor"])
        ax.spines["right"].set_color(params["axes.edgecolor"])
        ax.spines["left"].set_color(params["axes.edgecolor"])

        if title:
            ax.set_title(title, color=params["text.color"])
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(
            visible=True,
            color=params["grid.color"],
            alpha=params["grid.alpha"],
        )

        return fig, ax
