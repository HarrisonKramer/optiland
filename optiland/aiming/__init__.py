"""Ray Aiming.

This package provides classes for ray aiming.

Kramer Harrison, 2025
"""

from __future__ import annotations

from .base import AimingStrategy
from .factory import AimingStrategyFactory

__all__ = ["AimingStrategy", "AimingStrategyFactory"]
