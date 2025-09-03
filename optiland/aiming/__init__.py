"""Ray Aiming.

This package provides classes for ray aiming.

Kramer Harrison, 2025
"""

from __future__ import annotations

from .aiming import RayAiming
from .factory import RayAimingStrategyFactory
from .strategies.base import RayAimingStrategy

__all__ = ["RayAiming", "RayAimingStrategy", "RayAimingStrategyFactory"]
