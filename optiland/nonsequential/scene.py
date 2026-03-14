"""Non-sequential optical scene.

Top-level orchestrator that contains surfaces, sources, and a tracer.
Provides a high-level API for setting up and running non-sequential
ray traces.

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.nonsequential.detector import DetectorData
from optiland.nonsequential.tracer import NonSequentialTracer

if TYPE_CHECKING:
    from optiland.nonsequential.ray_data import NSQRayPool
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
        for source in self._sources:
            rays = source.generate_rays(n_rays)
            self._tracer.trace(
                self._surfaces,
                rays,
                self._detector_data,
            )

        return self._detector_data

    def trace_with_paths(self, n_rays: int = 10000) -> list[NSQRayPool]:
        """Trace rays from all sources, recording full path histories.

        Unlike ``trace()``, this method temporarily enables path recording on
        every source and returns the ray pools for visualization. Detector
        data is **not** accumulated (an empty dict is passed to the tracer).

        Sources without a ``record_paths`` attribute are traced without path
        recording and a warning is emitted; their returned pool will have
        ``path_history=None``.

        Args:
            n_rays: Number of rays per source.

        Returns:
            List of NSQRayPool instances (one per source) with
            ``path_history`` populated (or ``None`` for unsupported sources).
        """
        pools: list[NSQRayPool] = []

        for source in self._sources:
            if not hasattr(source, "record_paths"):
                warnings.warn(
                    f"Source {source!r} has no 'record_paths' attribute; "
                    "path recording skipped for this source.",
                    stacklevel=2,
                )
                pool = source.generate_rays(n_rays)
                self._tracer.trace(self._surfaces, pool, {})
                pools.append(pool)
                continue

            old_rp = source.record_paths
            source.record_paths = True
            try:
                pool = source.generate_rays(n_rays)
                # Snapshot source emission positions as sentinel step 0
                pool.record_path_point(be.full(n_rays, -1.0))
                self._tracer.trace(self._surfaces, pool, {})
            finally:
                source.record_paths = old_rp

            pools.append(pool)

        return pools

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
