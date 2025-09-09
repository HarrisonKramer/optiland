"""Thin-film multilayer calculations (Transfer Matrix Method).

Public API:
- ``Layer``: one thin-film layer (material + thickness)
- ``ThinFilmStack``: stack structure and TMM computations (r, t, R, T, A)

Units: wavelength in µm, thickness in µm (nm helpers), AOI in radians (deg helpers).
"""

from __future__ import annotations

from .analysis import SpectralAnalyzer
from .layer import Layer
from .stack import ThinFilmStack

__all__ = ["Layer", "ThinFilmStack", "SpectralAnalyzer"]
