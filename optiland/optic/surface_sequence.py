# optiland/optic/surface_sequence.py
"""Defines the SurfaceSequence class for custom ray tracing paths."""

from typing import List
from optiland.surfaces.standard_surface import Surface # Adjusted import
from optiland.rays.base import BaseRays # Adjusted import

class SurfaceSequence:
    """
    Represents a custom sequence of surfaces for ray tracing.

    This class allows defining an arbitrary path through surfaces,
    which may be a subset or a reordered version of surfaces from
    an Optic instance. It holds references to existing Surface objects
    to avoid data duplication.
    """
    def __init__(self, surfaces: List[Surface]):
        """
        Initializes the SurfaceSequence.

        Args:
            surfaces: An ordered list of Surface objects defining the
                      custom trace path.
        """
        self.surfaces: List[Surface] = surfaces

    def trace(self, rays: BaseRays):
        """
        Traces the given rays through this specific sequence of surfaces.

        This method will iterate through the `self.surfaces` list and
        sequentially call the `trace` method of each surface.

        Args:
            rays: The rays to be traced.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError(
            "SurfaceSequence.trace() is not yet implemented."
        )

    def __repr__(self):
        return f"<SurfaceSequence surfaces_count={len(self.surfaces)}>"
