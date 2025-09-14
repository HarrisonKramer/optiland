"""Log Scaler Module

This module contains the LogScaler class, which is a scaler that
performs a logarithmic transformation on the value.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be

from .base import Scaler


class LogScaler(Scaler):
    """Represents a scaler that performs a logarithmic transformation on the value."""

    def __init__(self, base=be.e):
        self.base = base

    def scale(self, value):
        """Scale the value using a logarithmic transformation.

        Args:
            value: The value to scale

        """
        return be.log(be.asarray(value)) / be.log(be.asarray(self.base))

    def inverse_scale(self, scaled_value):
        """Inverse scale the value using a logarithmic transformation.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return be.power(self.base, scaled_value)
