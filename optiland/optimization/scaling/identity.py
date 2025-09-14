"""Identity Scaler Module

This module contains the IdentityScaler class, which is a scaler that does
not perform any scaling.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.optimization.scaling.base import Scaler


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
