"""Base Scaler Module

This module contains the base class for a scaler in an optic system. The
Scaler class is an abstract class that represents the behavior of a
scaler in an optic system. It is used as a base class for all scalers in
the optimization process.

Kramer Harrison, 2024
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

    @abstractmethod
    def inverse_scale(self, scaled_value):
        """Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale

        """
        # pragma: no cover
