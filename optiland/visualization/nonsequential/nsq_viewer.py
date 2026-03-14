"""NSQ scene viewers: 2D (matplotlib) and 3D (VTK).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from optiland.visualization.nonsequential.base import BaseNSQViewer
from optiland.visualization.nonsequential.rays import NSQRays2D, NSQRays3D
from optiland.visualization.nonsequential.scene_renderer import (
    NSQSceneRenderer2D,
    NSQSceneRenderer3D,
)
from optiland.visualization.themes import get_active_theme

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NonSequentialScene


class NSQViewer(BaseNSQViewer):
    """2D matplotlib viewer for non-sequential optical scenes.

    Args:
        scene: The non-sequential scene to visualize.
        default_extent: Fallback surface extent (mm) for surfaces with no hits.
    """

    def __init__(self, scene: NonSequentialScene, default_extent: float = 5.0) -> None:
        super().__init__(scene)
        self.default_extent = default_extent

    def view(
        self,
        n_rays: int = 200,
        num_display_rays: int | None = 50,
        projection: str = "YZ",
        figsize=None,
        xlim=None,
        ylim=None,
        ax=None,
    ) -> tuple:
        """Render the non-sequential scene in 2D.

        Traces rays with path recording enabled, then plots surfaces and
        ray paths on a matplotlib figure.

        Args:
            n_rays: Number of rays to trace per source.
            num_display_rays: Number of rays to display (subsampled from
                ``n_rays``). ``None`` displays all traced rays.
            projection: Projection plane; one of ``'XY'``, ``'XZ'``, ``'YZ'``.
            figsize: Figure size tuple. ``None`` uses the active theme default.
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.
            ax: Existing matplotlib axis to plot on. ``None`` creates a new
                figure.

        Returns:
            Tuple of ``(fig, ax)``.
        """
        pools = self.scene.trace_with_paths(n_rays)

        theme = get_active_theme()
        params = theme.parameters
        if figsize is None:
            figsize = params.get("figure.figsize", (10, 4))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_facecolor(params.get("figure.facecolor", "white"))
        else:
            fig = ax.get_figure()

        ax.set_facecolor(params.get("axes.facecolor", "white"))

        NSQSceneRenderer2D(self.scene, pools, self.default_extent).plot(
            ax, theme=theme, projection=projection
        )
        NSQRays2D(self.scene).plot(
            ax, pools, theme=theme, projection=projection, num_rays=num_display_rays
        )

        ax.axis("image")
        if projection == "YZ":
            ax.set_xlabel("Z [mm]", color=params.get("axes.labelcolor", "black"))
            ax.set_ylabel("Y [mm]", color=params.get("axes.labelcolor", "black"))
        elif projection == "XZ":
            ax.set_xlabel("Z [mm]", color=params.get("axes.labelcolor", "black"))
            ax.set_ylabel("X [mm]", color=params.get("axes.labelcolor", "black"))
        else:  # XY
            ax.set_xlabel("X [mm]", color=params.get("axes.labelcolor", "black"))
            ax.set_ylabel("Y [mm]", color=params.get("axes.labelcolor", "black"))

        ax.tick_params(axis="x", colors=params.get("xtick.color", "black"))
        ax.tick_params(axis="y", colors=params.get("ytick.color", "black"))
        ax.grid(
            visible=True,
            color=params.get("grid.color", "gray"),
            alpha=params.get("grid.alpha", 0.3),
        )

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        return fig, ax


class NSQViewer3D(BaseNSQViewer):
    """3D VTK viewer for non-sequential optical scenes.

    Args:
        scene: The non-sequential scene to visualize.
        default_extent: Fallback surface extent (mm) for surfaces with no hits.
    """

    def __init__(self, scene: NonSequentialScene, default_extent: float = 5.0) -> None:
        import vtk

        super().__init__(scene)
        self.default_extent = default_extent
        self.ren_win = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

    def view(
        self,
        n_rays: int = 200,
        num_display_rays: int | None = 50,
        figsize: tuple[int, int] = (1200, 800),
        dark_mode: bool = False,
    ) -> None:
        """Render the non-sequential scene in 3D.

        Args:
            n_rays: Number of rays to trace per source.
            num_display_rays: Number of rays to display. ``None`` displays all.
            figsize: VTK render window size ``(width, height)``.
            dark_mode: If ``True``, uses a dark gradient background.
        """
        import vtk

        pools = self.scene.trace_with_paths(n_rays)

        renderer = vtk.vtkRenderer()
        self.ren_win.AddRenderer(renderer)
        self.iren.SetRenderWindow(self.ren_win)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        NSQSceneRenderer3D(self.scene, pools, self.default_extent).plot(renderer)
        NSQRays3D(self.scene).plot(renderer, pools, num_rays=num_display_rays)

        renderer.GradientBackgroundOn()
        renderer.SetGradientMode(vtk.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL)

        if dark_mode:
            renderer.SetBackground(0.13, 0.15, 0.19)
            renderer.SetBackground2(0.195, 0.21, 0.24)
        else:
            renderer.SetBackground(0.8, 0.9, 1.0)
            renderer.SetBackground2(0.4, 0.5, 0.6)

        self.ren_win.SetSize(*figsize)
        self.ren_win.SetWindowName("NSQ Scene - 3D Viewer")
        self.ren_win.Render()

        renderer.GetActiveCamera().SetPosition(1, 0, 0)
        renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        renderer.ResetCamera()
        renderer.GetActiveCamera().Elevation(0)
        renderer.GetActiveCamera().Azimuth(150)

        self.ren_win.Render()
        self.iren.Start()
