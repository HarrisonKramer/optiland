"""Identity Scaler Module

This module contains the IdentityScaler class, which is a "scaler" that does
not perform any scaling at all.

Kramer Harrison, 2025
"""

from __future__ import annotations

from .base import Scaler


class IdentityScaler(Scaler):
    """Represents a scaler that does not perform any scaling."""

    def scale(self, value):
        """Return the value without any scaling.

        Args:
            value: The value to scale

        """
        return value

    def inverse_scale(self, scaled_value):
        """Return the value without any inverse scaling.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return scaled_value
