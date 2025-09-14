"""Reciprocal Scaler Module

This module contains the ReciprocalScaler class, which is a scaler that
performs a reciprocal transformation on the value.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be

from .base import Scaler


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

    @property
    def monotonic(self) -> bool:
        """Returns True if the scaler is monotonic increasing, False otherwise."""
        return False

    def transform_bounds(self, min_val, max_val):
        """Transforms the bounds for a reciprocal scaler."""
        if min_val is not None and min_val > 0:
            if max_val is not None:
                return self.scale(max_val), self.scale(min_val)
            else:
                return 0, self.scale(min_val)
        elif max_val is not None and max_val < 0:
            if min_val is not None:
                return self.scale(min_val), self.scale(max_val)
            else:
                return self.scale(max_val), 0

        # This scaler is not defined for bounds that cross zero
        if (min_val is not None and min_val <= 0) and (
            max_val is not None and max_val >= 0
        ):
            raise ValueError("Reciprocal scaler bounds cannot cross zero.")

        return super().transform_bounds(min_val, max_val)
