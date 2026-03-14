"""Non-sequential optical scene.

Top-level orchestrator that contains surfaces, sources, and a tracer.
Provides a high-level API for setting up and running non-sequential
ray traces.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.nonsequential.detector import DetectorData
from optiland.nonsequential.tracer import NonSequentialTracer

if TYPE_CHECKING:
    from optiland.nonsequential.source import BaseSource
    from optiland.nonsequential.surface import NSQSurface


class NonSequentialScene:
    """A non-sequential optical scene.

    Contains surfaces, sources, and a tracer. Provides a high-level
    API for setting up and running non-sequential ray traces.

    Args:
        intensity_threshold: Minimum ray intensity before termination.
        max_interactions: Maximum number of surface interactions per trace.
    """

    def __init__(
        self,
        intensity_threshold: float = 1e-6,
        max_interactions: int = 100,
    ):
        self._surfaces: list[NSQSurface] = []
        self._sources: list[BaseSource] = []
        self._tracer = NonSequentialTracer(intensity_threshold, max_interactions)
        self._detector_data: dict[int, DetectorData] = {}
        self._next_id = 0
        self._last_traced_rays: list = []

    def add_surface(self, surface: NSQSurface) -> None:
        """Add a surface to the scene. Assigns a unique surface_id.

        Args:
            surface: The surface to add.
        """
        surface.surface_id = self._next_id
        self._next_id += 1
        self._surfaces.append(surface)

        if surface.is_detector:
            self._detector_data[surface.surface_id] = DetectorData()

    def add_source(self, source: BaseSource) -> None:
        """Add a light source to the scene.

        Args:
            source: The source to add.
        """
        self._sources.append(source)

    def trace(self, n_rays: int = 10000) -> dict[int, DetectorData]:
        """Trace rays from all sources through the scene.

        Accumulates results on detector surfaces. Call multiple times
        for Monte Carlo convergence.

        Args:
            n_rays: Number of rays per source.

        Returns:
            Dictionary mapping detector surface_id to DetectorData.
        """
        self._last_traced_rays = []
        for source in self._sources:
            rays = source.generate_rays(n_rays)
            self._tracer.trace(
                self._surfaces,
                rays,
                self._detector_data,
            )
            if rays.record_paths:
                self._last_traced_rays.append(rays)

        return self._detector_data

    def reset_detectors(self) -> None:
        """Clear all accumulated detector data."""
        for detector in self._detector_data.values():
            detector.reset()

    @property
    def surfaces(self) -> tuple[NSQSurface, ...]:
        """All surfaces in the scene."""
        return tuple(self._surfaces)

    @property
    def detector_surfaces(self) -> tuple[NSQSurface, ...]:
        """Surfaces flagged as detectors."""
        return tuple(s for s in self._surfaces if s.is_detector)

    def draw(
        self,
        figsize: tuple | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        title: str | None = None,
        projection: str = "YZ",
        ax=None,
    ) -> tuple:
        """Generate a 2D cross-sectional view of the non-sequential scene.

        Args:
            figsize: Figure size.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            title: Plot title.
            projection: Projection plane ('YZ', 'XZ', 'XY').
            ax: Optional matplotlib axes to plot on.

        Returns:
            Tuple of (figure, axes, viewer).
        """
        from optiland.visualization.nonsequential.viewer import NSQViewer

        viewer = NSQViewer(self)
        fig, ax = viewer.view(
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            title=title,
            projection=projection,
            ax=ax,
        )
        return fig, ax, viewer

    def draw3d(
        self,
        figsize: tuple = (1200, 800),
        dark_mode: bool = False,
    ) -> None:
        """Generate an interactive 3D view of the non-sequential scene.

        Args:
            figsize: Window size.
            dark_mode: Whether to use a dark theme.
        """
        from optiland.visualization.nonsequential.viewer_3d import NSQViewer3D

        viewer = NSQViewer3D(self)
        viewer.view(
            figsize=figsize,
            dark_mode=dark_mode,
        )

    def get_detector_data(self, surface_id: int) -> DetectorData:
        """Get detector data for a specific surface.

        Args:
            surface_id: The surface ID of the detector.

        Returns:
            The accumulated detector data.

        Raises:
            KeyError: If the surface_id is not a detector.
        """
        return self._detector_data[surface_id]
