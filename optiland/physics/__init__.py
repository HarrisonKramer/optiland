"""Physics kernels for optical computations.

This package provides standalone physics functions (refraction, reflection)
that can be shared between sequential and non-sequential ray tracers.

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland.physics.interaction import reflect, refract

__all__ = ["refract", "reflect"]
