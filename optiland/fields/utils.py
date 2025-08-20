"""Fields Utility Functions Module

This module provides utility functions for managing field properties in optical systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from optiland.fields.field_modes import BaseFieldMode
    from optiland.optic import Optic


@contextmanager
def temporary_field_mode(optic: Optic, temp_mode: BaseFieldMode):
    """Temporarily override a read-only property with a fixed value."""
    original = optic.__class__.field_mode
    try:
        optic.field_mode = property(lambda self: temp_mode)
        yield
    finally:
        # Restore the original property
        optic.field_mode = original


@dataclass(frozen=True)
class ResolvedField:
    """Canonical field definition resolved from an image height request.

    Attributes:
        mode_type: "angle" when object is at infinity, "object_height" otherwise.
        value: Canonical field value (degrees for angle, length units for height).
    """

    mode_type: Literal["angle", "object_height"]
    value: float
