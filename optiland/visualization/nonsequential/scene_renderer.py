"""Non-sequential scene surface renderer.

Provides NSQSceneRenderer2D and NSQSceneRenderer3D for rendering
NSQSurface objects using the Surface2D/3D visualization classes.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.visualization.nonsequential.adapter import NSQSurfaceAdapter
from optiland.visualization.system.surface import Surface2D, Surface3D
from optiland.visualization.system.utils import transform

if TYPE_CHECKING:
    from optiland.nonsequential.ray_data import NSQRayPool
    from optiland.nonsequential.scene import NonSequentialScene


class NSQSceneRenderer2D:
    """Renders NSQ scene surfaces in 2D using matplotlib.

    Computes per-surface extents from ray path histories and delegates
    drawing to :class:`Surface2D` instances.

    Args:
        scene: The non-sequential scene whose surfaces to render.
        pools: Ray pools with ``path_history`` populated.
        default_extent: Fallback extent (mm) for surfaces with no ray hits.
    """

    def __init__(
        self,
        scene: NonSequentialScene,
        pools: list[NSQRayPool],
        default_extent: float = 5.0,
    ) -> None:
        self.scene = scene
        self.pools = pools
        self.default_extent = default_extent

    def plot(self, ax, theme=None, projection: str = "YZ") -> None:
        """Plot all surfaces on a matplotlib axis.

        Args:
            ax: Matplotlib axis to plot on.
            theme: Optional theme for styling.
            projection: Projection plane; one of ``'XY'``, ``'XZ'``, ``'YZ'``.
        """
        extent_map = self._build_extent_map()
        for surface in self.scene.surfaces:
            extent = extent_map.get(surface.surface_id, self.default_extent)
            adapter = NSQSurfaceAdapter(surface)
            surf2d = Surface2D(adapter, extent)
            surf2d.plot(ax, theme=theme, projection=projection)

    def _build_extent_map(self) -> dict[int, float]:
        """Compute the maximum ray radius on each surface from path histories.

        For each path_history step (skipping the sentinel step 0 with
        ``surface_ids == -1``), collects ray positions per surface and
        transforms them to local coordinates to find the maximum radial
        extent.

        Returns:
            Mapping from surface_id to maximum radial extent in local coords.
        """
        surf_lookup = {s.surface_id: s for s in self.scene.surfaces}
        extent_map: dict[int, float] = {}

        for pool in self.pools:
            if pool.path_history is None or len(pool.path_history) < 2:
                continue

            # Skip step 0 (sentinel emission: surface_ids all == -1)
            for step in pool.path_history[1:]:
                x_arr = be.to_numpy(step["x"])
                y_arr = be.to_numpy(step["y"])
                z_arr = be.to_numpy(step["z"])
                sid_arr = be.to_numpy(step["surface_ids"]).astype(int)
                active_arr = be.to_numpy(step["active"])

                unique_sids = set(sid_arr[active_arr])
                for sid in unique_sids:
                    if sid == -1 or sid not in surf_lookup:
                        continue
                    mask = active_arr & (sid_arr == sid)
                    xs = be.array(x_arr[mask])
                    ys = be.array(y_arr[mask])
                    zs = be.array(z_arr[mask])

                    adapter = NSQSurfaceAdapter(surf_lookup[sid])
                    xl, yl, _ = transform(xs, ys, zs, adapter, is_global=True)
                    r = np.hypot(be.to_numpy(xl), be.to_numpy(yl))
                    if len(r) == 0:
                        continue
                    r_max = float(np.max(r))
                    current = extent_map.get(sid, 0.0)
                    extent_map[sid] = max(current, r_max)

        return extent_map


class NSQSceneRenderer3D(NSQSceneRenderer2D):
    """Renders NSQ scene surfaces in 3D using VTK.

    Extends :class:`NSQSceneRenderer2D` to produce VTK actors via
    :class:`Surface3D`.

    Args:
        scene: The non-sequential scene whose surfaces to render.
        pools: Ray pools with ``path_history`` populated.
        default_extent: Fallback extent for surfaces with no ray hits.
    """

    def plot(self, renderer, theme=None) -> None:  # type: ignore[override]
        """Add surface actors to a VTK renderer.

        Args:
            renderer: VTK renderer to add actors to.
            theme: Optional theme for styling.
        """
        extent_map = self._build_extent_map()
        for surface in self.scene.surfaces:
            extent = extent_map.get(surface.surface_id, self.default_extent)
            adapter = NSQSurfaceAdapter(surface)
            surf3d = Surface3D(adapter, extent)
            surf3d.plot(renderer, theme=theme)
