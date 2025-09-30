"""Base Scaler Module

This module contains the base class for scaling variables in an optic system. Scaling is
used in optimization to improve convergence and performance.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Scaler(ABC):
    """Represents the behavior of a scaler in an optic system."""

    @abstractmethod
    def scale(self, value):
        """Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale

        """
        # pragma: no cover

    def transform_bounds(self, min_val, max_val):
        """Transforms the bounds according to the scaler's behavior."""
        if not self.monotonic:
            min_val, max_val = max_val, min_val

        scaled_min = self.scale(min_val) if min_val is not None else None
        scaled_max = self.scale(max_val) if max_val is not None else None
        return scaled_min, scaled_max

    @property
    def monotonic(self) -> bool:
        """Returns True if the scaler is monotonic increasing, False otherwise."""
        return True

    @abstractmethod
    def inverse_scale(self, scaled_value):
        """Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        # pragma: no cover
