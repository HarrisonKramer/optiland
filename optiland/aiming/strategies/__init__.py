"""Ray Aiming Strategies.

This package provides various strategies for ray aiming.

Kramer Harrison, 2025
"""

from __future__ import annotations

from .base import RayAimingStrategy
from .cached import CachedAimingStrategy
from .fallback import FallbackAimingStrategy
from .iterative import IterativeAimingStrategy
from .model_based import ModelBasedAimingStrategy
from .paraxial import ParaxialAimingStrategy

__all__ = [
    "RayAimingStrategy",
    "ParaxialAimingStrategy",
    "IterativeAimingStrategy",
    "FallbackAimingStrategy",
    "CachedAimingStrategy",
    "ModelBasedAimingStrategy",
]
