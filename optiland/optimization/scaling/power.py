"""Power Scaler Module

This module contains the PowerScaler class, which is a scaler that
performs a power transformation on the value.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be

from .base import Scaler


class PowerScaler(Scaler):
    """Represents a scaler that performs a power transformation on the value."""

    def __init__(self, power=1.0):
        self.power = power

    def scale(self, value):
        """Scale the value using a power transformation.

        Args:
            value: The value to scale

        """
        return be.power(value, self.power)

    def inverse_scale(self, scaled_value):
        """Inverse scale the value using a power transformation.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        return be.power(scaled_value, 1.0 / self.power)
