"""Non-sequential Rays Visualization Module

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import vtk

import optiland.backend as be
from optiland.visualization.system.ray_bundle import RayBundle


class NSQRays2D:
    """A class to represent and visualize 2D rays in a non-sequential scene.

    Args:
        scene (NonSequentialScene): The scene to be visualized.
    """

    def __init__(self, scene):
        self.scene = scene

    def plot(
        self,
        ax,
        theme=None,
        projection="YZ",
        hide_vignetted=False,
    ):
        """Plots the traced rays.

        Args:
            ax: The matplotlib axis to plot on.
            theme (Theme, optional): The theme to apply. Defaults to None.
            projection (str, optional): The projection plane. Must be 'XY',
                'XZ', or 'YZ'. Defaults to 'YZ'.
            hide_vignetted (bool, optional): If True, rays that vignette at any
                surface are not shown. Defaults to False.
        """
        artists = {}
        for rays in self.scene._last_traced_rays:
            if not rays.record_paths or rays.path_history is None or len(rays.path_history) < 2:
                continue

            n_rays = len(rays.x)
            n_steps = len(rays.path_history)

            x = np.zeros((n_steps, n_rays))
            y = np.zeros((n_steps, n_rays))
            z = np.zeros((n_steps, n_rays))
            active = np.zeros((n_steps, n_rays), dtype=bool)

            for step_idx, step_data in enumerate(rays.path_history):
                x[step_idx, :] = be.to_numpy(step_data["x"])
                y[step_idx, :] = be.to_numpy(step_data["y"])
                z[step_idx, :] = be.to_numpy(step_data["z"])
                active[step_idx, :] = be.to_numpy(step_data["active"])

            for k in range(n_rays):
                ray_active = active[:, k]
                if hide_vignetted and not ray_active[-1]:
                    # Ray was terminated early (e.g. vignetted or below threshold)
                    continue

                # Find out how many segments are valid
                valid_indices = np.where(ray_active)[0]
                if len(valid_indices) == 0:
                    continue

                # Including the step right after it becomes inactive to draw the ray to the surface
                last_valid = valid_indices[-1]
                if last_valid + 1 < n_steps:
                    valid_indices = np.append(valid_indices, last_valid + 1)

                xk = x[valid_indices, k]
                yk = y[valid_indices, k]
                zk = z[valid_indices, k]

                color_idx = 0  # Could be mapped to source or wavelength later

                artist, ray_bundle = self._plot_single_line(
                    ax, xk, yk, zk, color_idx, field=(0,0), theme=theme, projection=projection
                )
                artists[artist] = ray_bundle

        return artists

    def _plot_single_line(
        self, ax, x, y, z, color_idx, field, linewidth=1, theme=None, projection="YZ"
    ):
        if theme:
            ray_cycle = theme.parameters.get("ray_cycle")
            color = ray_cycle[color_idx % len(ray_cycle)]
        else:
            color = f"C{color_idx}"

        if projection == "XY":
            (line,) = ax.plot(x, y, color=color, linewidth=linewidth)
        elif projection == "XZ":
            (line,) = ax.plot(z, x, color=color, linewidth=linewidth)
        else:  # YZ
            (line,) = ax.plot(z, y, color=color, linewidth=linewidth)
        return line, RayBundle(x, y, z, field)


class NSQRays3D(NSQRays2D):
    """A class to represent 3D rays for visualization using VTK in non-sequential scenes.
    """

    def __init__(self, scene):
        super().__init__(scene)
        self._rgb_colors = [
            (0.122, 0.467, 0.706),
            (1.000, 0.498, 0.055),
            (0.173, 0.627, 0.173),
            (0.839, 0.153, 0.157),
            (0.580, 0.404, 0.741),
            (0.549, 0.337, 0.294),
            (0.890, 0.467, 0.761),
            (0.498, 0.498, 0.498),
            (0.737, 0.741, 0.133),
            (0.090, 0.745, 0.812),
        ]

    def plot(
        self,
        renderer,
        theme=None,
        hide_vignetted=False,
    ):
        """Plots the traced rays in 3D.
        """
        for rays in self.scene._last_traced_rays:
            if not rays.record_paths or rays.path_history is None or len(rays.path_history) < 2:
                continue

            n_rays = len(rays.x)
            n_steps = len(rays.path_history)

            x = np.zeros((n_steps, n_rays))
            y = np.zeros((n_steps, n_rays))
            z = np.zeros((n_steps, n_rays))
            active = np.zeros((n_steps, n_rays), dtype=bool)

            for step_idx, step_data in enumerate(rays.path_history):
                x[step_idx, :] = be.to_numpy(step_data["x"])
                y[step_idx, :] = be.to_numpy(step_data["y"])
                z[step_idx, :] = be.to_numpy(step_data["z"])
                active[step_idx, :] = be.to_numpy(step_data["active"])

            for k in range(n_rays):
                ray_active = active[:, k]
                if hide_vignetted and not ray_active[-1]:
                    continue

                valid_indices = np.where(ray_active)[0]
                if len(valid_indices) == 0:
                    continue

                last_valid = valid_indices[-1]
                if last_valid + 1 < n_steps:
                    valid_indices = np.append(valid_indices, last_valid + 1)

                xk = x[valid_indices, k]
                yk = y[valid_indices, k]
                zk = z[valid_indices, k]

                color_idx = 0

                self._plot_single_line(
                    renderer, xk, yk, zk, color_idx, field=(0,0), theme=theme
                )

    def _plot_single_line(
        self, renderer, x, y, z, color_idx, field, linewidth=1, theme=None
    ):
        if theme:
            from matplotlib.colors import to_rgb
            ray_cycle = theme.parameters.get("ray_cycle")
            color = to_rgb(ray_cycle[color_idx % len(ray_cycle)])
        else:
            color = self._rgb_colors[color_idx % 10]

        for k in range(1, len(x)):
            p0 = [x[k - 1], y[k - 1], z[k - 1]]
            p1 = [x[k], y[k], z[k]]

            line_source = vtk.vtkLineSource()
            line_source.SetPoint1(p0)
            line_source.SetPoint2(p1)

            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInputConnection(line_source.GetOutputPort())
            line_actor = vtk.vtkActor()
            line_actor.SetMapper(line_mapper)
            line_actor.GetProperty().SetLineWidth(linewidth)
            line_actor.GetProperty().SetColor(color)

            renderer.AddActor(line_actor)
