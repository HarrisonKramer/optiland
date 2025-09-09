"""Open-source Optical Design in Python"""

from __future__ import annotations

__version__ = "0.5.6"

# Import the sources for easy access
from optiland.sources import (
    BaseSource,
    CollimatedGaussianSource,
    GaussianSource,
)

__all__ = [
    "BaseSource",
    "CollimatedGaussianSource",
    "GaussianSource",
]
