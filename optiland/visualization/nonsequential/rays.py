"""Non-sequential ray path visualization.

Provides NSQRays2D and NSQRays3D for rendering ray path histories
recorded during non-sequential tracing.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.nonsequential.ray_data import NSQRayPool
    from optiland.nonsequential.scene import NonSequentialScene


# matplotlib default colors converted to RGB (mirrors Rays3D._rgb_colors)
_RGB_COLORS = [
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


class NSQRays2D:
    """Renders 2D ray paths from non-sequential trace path histories.

    Args:
        scene: The non-sequential scene (retained for future extensibility).
    """

    def __init__(self, scene: NonSequentialScene) -> None:
        self.scene = scene

    def plot(
        self,
        ax,
        pools: list[NSQRayPool],
        theme=None,
        projection: str = "YZ",
        num_rays: int | None = None,
    ) -> None:
        """Plot ray paths on a matplotlib axis.

        Iterates through path_history snapshots in each pool and draws line
        segments connecting consecutive active positions for each ray.

        Duplicate consecutive positions (an artifact of the tracer calling
        ``record_path_point`` once per hit surface per iteration) produce
        zero-length invisible segments and are harmless.

        Args:
            ax: Matplotlib axis to plot on.
            pools: List of NSQRayPool instances with ``path_history`` set.
            theme: Optional theme for color cycling.
            projection: Projection plane; one of ``'XY'``, ``'XZ'``, ``'YZ'``.
            num_rays: If given, subsample to at most this many display rays.
        """
        for source_idx, pool in enumerate(pools):
            if pool.path_history is None or len(pool.path_history) < 2:
                continue

            n_total = len(be.to_numpy(pool.x))
            if num_rays is not None:
                step_size = max(1, n_total // num_rays)
                ray_indices = range(0, n_total, step_size)
            else:
                ray_indices = range(n_total)

            if theme:
                ray_cycle = theme.parameters.get("ray_cycle", [])
                color = (
                    ray_cycle[source_idx % len(ray_cycle)]
                    if ray_cycle
                    else f"C{source_idx}"
                )
            else:
                color = f"C{source_idx}"

            for k in ray_indices:
                xs: list[float] = []
                ys: list[float] = []
                zs: list[float] = []
                for step_data in pool.path_history:
                    active_arr = be.to_numpy(step_data["active"])
                    if not bool(active_arr[k]):
                        continue
                    xs.append(float(be.to_numpy(step_data["x"])[k]))
                    ys.append(float(be.to_numpy(step_data["y"])[k]))
                    zs.append(float(be.to_numpy(step_data["z"])[k]))

                if len(xs) < 2:
                    continue

                x_np = np.array(xs)
                y_np = np.array(ys)
                z_np = np.array(zs)

                if projection == "XY":
                    ax.plot(x_np, y_np, color=color, linewidth=0.5, alpha=0.6)
                elif projection == "XZ":
                    ax.plot(z_np, x_np, color=color, linewidth=0.5, alpha=0.6)
                else:  # YZ
                    ax.plot(z_np, y_np, color=color, linewidth=0.5, alpha=0.6)


class NSQRays3D(NSQRays2D):
    """Renders 3D ray paths using VTK line actors.

    Extends NSQRays2D to produce VTK actors for 3D visualization.

    Args:
        scene: The non-sequential scene.
    """

    def plot(  # type: ignore[override]
        self,
        renderer,
        pools: list[NSQRayPool],
        theme=None,
        num_rays: int | None = None,
    ) -> None:
        """Add ray path actors to a VTK renderer.

        Args:
            renderer: VTK renderer to add actors to.
            pools: List of NSQRayPool instances with ``path_history`` set.
            theme: Optional theme for color cycling.
            num_rays: If given, subsample to at most this many display rays.
        """
        import vtk

        for source_idx, pool in enumerate(pools):
            if pool.path_history is None or len(pool.path_history) < 2:
                continue

            n_total = len(be.to_numpy(pool.x))
            if num_rays is not None:
                step_size = max(1, n_total // num_rays)
                ray_indices = range(0, n_total, step_size)
            else:
                ray_indices = range(n_total)

            if theme:
                from matplotlib.colors import to_rgb

                ray_cycle = theme.parameters.get("ray_cycle", [])
                color = (
                    to_rgb(ray_cycle[source_idx % len(ray_cycle)])
                    if ray_cycle
                    else _RGB_COLORS[source_idx % len(_RGB_COLORS)]
                )
            else:
                color = _RGB_COLORS[source_idx % len(_RGB_COLORS)]

            for k in ray_indices:
                xs: list[float] = []
                ys: list[float] = []
                zs: list[float] = []
                for step_data in pool.path_history:
                    active_arr = be.to_numpy(step_data["active"])
                    if not bool(active_arr[k]):
                        continue
                    xs.append(float(be.to_numpy(step_data["x"])[k]))
                    ys.append(float(be.to_numpy(step_data["y"])[k]))
                    zs.append(float(be.to_numpy(step_data["z"])[k]))

                if len(xs) < 2:
                    continue

                for i in range(1, len(xs)):
                    p0 = [xs[i - 1], ys[i - 1], zs[i - 1]]
                    p1 = [xs[i], ys[i], zs[i]]

                    line_source = vtk.vtkLineSource()
                    line_source.SetPoint1(p0)
                    line_source.SetPoint2(p1)

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(line_source.GetOutputPort())

                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetLineWidth(1)
                    actor.GetProperty().SetColor(color)

                    renderer.AddActor(actor)
