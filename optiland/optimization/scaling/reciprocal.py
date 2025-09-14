"""Reciprocal Scaler Module

This module contains the ReciprocalScaler class, which is a scaler that
performs a reciprocal transformation on the value.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.optimization.scaling.base import Scaler


class ReciprocalScaler(Scaler):
    """Represents a scaler that performs a reciprocal transformation on the value."""

    def scale(self, value):
        """Scale the value using a reciprocal transformation.

        Args:
            value: The value to scale

        """
        if value != 0:
            return 1.0 / value if be.isfinite(be.asarray(value)) else 0.0
        else:
            return be.inf

    def inverse_scale(self, scaled_value):
        """Inverse scale the value using a reciprocal transformation.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        if scaled_value != 0:
            return 1.0 / scaled_value if be.isfinite(be.asarray(scaled_value)) else 0.0
        else:
            return be.inf
