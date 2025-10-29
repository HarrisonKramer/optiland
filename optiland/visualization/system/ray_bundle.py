"""Ray Bundle Module

This module provides a simple container for ray bundle data.

"""

from __future__ import annotations


class RayBundle:
    """A simple container for ray bundle data."""

    def __init__(self, x, y, z, bundle_id=None):
        self.x = x
        self.y = y
        self.z = z
        self.bundle_id = bundle_id
