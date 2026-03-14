"""Non-sequential Scene Visualization Module (3D).

Kramer Harrison, 2026
"""

from __future__ import annotations

import vtk

from optiland.visualization.base import BaseViewer
from optiland.visualization.nonsequential.surface import NSQSurface3D
from optiland.visualization.nonsequential.rays import NSQRays3D


class NSQViewer3D(BaseViewer):
    """A class used to visualize non-sequential scenes in 3D.

    Args:
        scene: The non-sequential scene to be visualized.
    """

    def __init__(self, scene):
        self.scene = scene
        self.rays_viewer = NSQRays3D(scene)
        self.ren_win = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

    def view(
        self,
        figsize=(1200, 800),
        dark_mode=False,
    ):
        """Visualizes the non-sequential scene in 3D.

        Args:
            figsize (tuple, optional): The size of the figure.
                Defaults to (1200, 800).
            dark_mode (bool, optional): Whether to use dark mode.
                Defaults to False.
        """
        renderer = vtk.vtkRenderer()
        self.ren_win.AddRenderer(renderer)

        self.iren.SetRenderWindow(self.ren_win)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        # Plot surfaces
        for surf in self.scene.surfaces:
            surf_viewer = NSQSurface3D(surf)
            surf_viewer.plot(renderer)

        # Plot rays
        self.rays_viewer.plot(renderer)

        renderer.GradientBackgroundOn()
        renderer.SetGradientMode(vtk.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL)

        if dark_mode:
            renderer.SetBackground(0.13, 0.15, 0.19)
            renderer.SetBackground2(0.195, 0.21, 0.24)
        else:
            renderer.SetBackground(0.8, 0.9, 1.0)
            renderer.SetBackground2(0.4, 0.5, 0.6)

        self.ren_win.SetSize(*figsize)
        self.ren_win.SetWindowName("Non-Sequential Scene - 3D Viewer")
        self.ren_win.Render()

        renderer.GetActiveCamera().SetPosition(1, 0, 0)
        renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        renderer.GetActiveCamera().SetViewUp(0, 1, 0)
        renderer.ResetCamera()
        renderer.GetActiveCamera().Elevation(0)
        renderer.GetActiveCamera().Azimuth(150)

        self.ren_win.Render()
        self.iren.Start()
