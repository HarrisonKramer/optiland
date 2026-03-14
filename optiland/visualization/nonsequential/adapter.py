"""NSQ surface adapter for sequential Surface2D/3D visualization.

Adapts an NSQSurface to the minimal interface expected by Surface2D and
Surface3D: ``.geometry``, ``.aperture`` (always None), and ``.comment``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.surface import NSQSurface


class NSQSurfaceAdapter:
    """Makes NSQSurface compatible with Surface2D/3D.

    Surface2D and Surface3D access ``surf.geometry``, ``surf.aperture``,
    and ``surf.comment``.  With ``aperture=None`` the sequential classes fall
    back to the ``ray_extent`` argument for sizing and skip all aperture
    clipping logic.

    Args:
        nsq_surface: The non-sequential surface to adapt.
    """

    def __init__(self, nsq_surface: NSQSurface) -> None:
        self.geometry = nsq_surface.geometry
        self.aperture = None
        self.comment = (
            nsq_surface.label
            if nsq_surface.label
            else f"surface_{nsq_surface.surface_id}"
        )
