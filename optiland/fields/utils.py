"""Fields Utility Functions Module

This module provides utility functions for managing field properties in optical systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def override_property(obj, property_name, temp_value):
    """Temporarily override a read-only property with a fixed value."""
    cls = type(obj)
    original = getattr(cls, property_name)
    try:
        # Replace the property with a temporary one that returns temp_value
        setattr(cls, property_name, property(lambda self: temp_value))
        yield
    finally:
        # Restore the original property
        setattr(cls, property_name, original)
