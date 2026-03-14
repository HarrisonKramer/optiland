"""Non-sequential Surface Visualization Module

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import vtk

import optiland.backend as be
from optiland.rays import RealRays
from optiland.visualization.system.utils import transform, transform_3d


class NSQSurface2D:
    """A class used to represent a 2D surface for non-sequential visualization.

    Args:
        surf (NSQSurface): The non-sequential surface object.
        extent (float): The default extent to plot.
    """

    def __init__(self, surface, extent=10.0):
        self.surf = surface
        # Try to infer extent from aperture if any, else use a fallback.
        # Note: NSQSurface does not yet have explicit aperture properties in current API,
        # but if the geometry does, we might use it. For now, we use a fixed fallback or a bounding box.
        self.extent = extent

    def plot(self, ax, theme=None, projection="YZ"):
        """Plots the surface on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis.
            theme (Theme, optional): The theme to use. Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY', 'XZ', or 'YZ'.
        """
        x, y, z = self._compute_sag(projection)

        x = be.to_numpy(x)
        y = be.to_numpy(y)
        z = be.to_numpy(z)

        # Draw line for the surface
        if theme:
            color = theme.parameters.get("axes.edgecolor")
        else:
            color = "black"

        if projection == "XY":
            (line,) = ax.plot(x, y, color=color, linewidth=1.0)
        elif projection == "XZ":
            (line,) = ax.plot(z, x, color=color, linewidth=1.0)
        else:  # YZ
            (line,) = ax.plot(z, y, color=color, linewidth=1.0)

        return {line: self.surf}

    def _compute_sag(self, projection):
        # Create local coordinates
        r = be.linspace(-self.extent, self.extent, 201)
        zero_arr = be.zeros_like(r)

        if projection == "XY":
            # For XY projection, we just draw a circle or box.
            # But normally surfaces in sequential XY show full aperture face.
            # In non-sequential cross section, XY might be flat. Let's just do a circle.
            theta = be.linspace(0, 2 * np.pi, 201)
            x_local = self.extent * be.cos(theta)
            y_local = self.extent * be.sin(theta)

            rays = RealRays(
                x_local, y_local, be.zeros_like(x_local),
                be.zeros_like(x_local), be.zeros_like(x_local), be.ones_like(x_local),
                be.ones_like(x_local), be.ones_like(x_local)
            )
            z_local = self.surf.geometry.sag(x_local, y_local)

            x_global, y_global, z_global = transform(x_local, y_local, z_local, self.surf, is_global=False)
            return x_global, y_global, z_global

        if projection == "YZ":
            x_local = zero_arr
            y_local = r
        else:  # XZ
            x_local = r
            y_local = zero_arr

        rays = RealRays(
            x_local, y_local, be.zeros_like(x_local),
            be.zeros_like(x_local), be.zeros_like(x_local), be.ones_like(x_local),
            be.ones_like(x_local), be.ones_like(x_local)
        )
        z_local = self.surf.geometry.sag(x_local, y_local)

        x_global, y_global, z_global = transform(x_local, y_local, z_local, self.surf, is_global=False)
        return x_global, y_global, z_global


class NSQSurface3D(NSQSurface2D):
    """A class used to represent a 3D surface for non-sequential visualization using VTK.
    """

    def __init__(self, surface, extent=10.0):
        super().__init__(surface, extent)

    def plot(self, renderer, color=(0.8, 0.8, 0.8), opacity=0.3):
        """Plots the surface in 3D using VTK.

        Args:
            renderer (vtkRenderer): The VTK renderer.
            color (tuple, optional): RGB color tuple.
            opacity (float, optional): Opacity.
        """
        grid_size = 50
        r_points = be.linspace(-self.extent, self.extent, grid_size)
        x_local, y_local = be.meshgrid(r_points, r_points)
        x_local = x_local.flatten()
        y_local = y_local.flatten()

        # mask circle
        r_sq = x_local**2 + y_local**2
        mask = r_sq <= self.extent**2
        x_local = x_local[mask]
        y_local = y_local[mask]

        rays = RealRays(
            x_local, y_local, be.zeros_like(x_local),
            be.zeros_like(x_local), be.zeros_like(x_local), be.ones_like(x_local),
            be.ones_like(x_local), be.ones_like(x_local)
        )
        z_local = self.surf.geometry.sag(x_local, y_local)

        x_global, y_global, z_global = transform(x_local, y_local, z_local, self.surf, is_global=False)

        x_np = be.to_numpy(x_global)
        y_np = be.to_numpy(y_global)
        z_np = be.to_numpy(z_global)

        points = vtk.vtkPoints()
        for i in range(len(x_np)):
            points.InsertNextPoint(x_np[i], y_np[i], z_np[i])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(polydata)
        delaunay.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(delaunay.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)

        # Make front/back side visible
        actor.GetProperty().SetFrontfaceCulling(False)
        actor.GetProperty().SetBackfaceCulling(False)

        renderer.AddActor(actor)
