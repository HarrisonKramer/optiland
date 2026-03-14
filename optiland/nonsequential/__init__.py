"""Non-sequential ray tracing for optiland.

This package provides non-sequential ray tracing capabilities, allowing
rays to interact with surfaces in any order, follow physically determined
paths, split at partially reflective interfaces, and terminate based on
energy thresholds.

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland.nonsequential.detector import DetectorData
from optiland.nonsequential.ray_data import NSQRayPool
from optiland.nonsequential.scene import NonSequentialScene
from optiland.nonsequential.source import BaseSource, PointSource
from optiland.nonsequential.surface import NSQSurface
from optiland.nonsequential.tracer import NonSequentialTracer

__all__ = [
    "DetectorData",
    "NSQRayPool",
    "NSQSurface",
    "NonSequentialScene",
    "NonSequentialTracer",
    "BaseSource",
    "PointSource",
]
