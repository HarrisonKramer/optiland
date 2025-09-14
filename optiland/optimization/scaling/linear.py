"""Linear Scaler Module

This module contains the LinearScaler class, which is a scaler that
performs a linear transformation on the value.

Kramer Harrison, 2025
"""

from __future__ import annotations

from .base import Scaler


class LinearScaler(Scaler):
    """Represents a scaler that performs a linear transformation on the value."""

    def __init__(self, factor=1.0, offset=0.0):
        self.factor = factor
        self.offset = offset

    def scale(self, value):
        """Scale the value using a linear transformation.

        Args:
            value: The value to scale

        """
        return value * self.factor + self.offset

    def inverse_scale(self, scaled_value):
        """Inverse scale the value using a linear transformation.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return (scaled_value - self.offset) / self.factor
