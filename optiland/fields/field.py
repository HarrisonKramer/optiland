"""Field Module

This module defines the Field class, which represents a field in an optical system.

Kramer Harrison, 2025
"""

from __future__ import annotations


class Field:
    """Represents a field with specific properties.

    Attributes:
        x (int): The x-coordinate of the field.
        y (int): The y-coordinate of the field.
        vx (float): The vignette factor in the x-direction.
        vy (float): The vignette factor in the y-direction.

    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        vignette_factor_x: float = 0.0,
        vignette_factor_y: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.vx = vignette_factor_x
        self.vy = vignette_factor_y

    def to_dict(self) -> dict:
        """Convert the field to a dictionary.

        Returns:
            A dictionary representation of the field.

        """
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
        }

    @classmethod
    def from_dict(cls, field_dict: dict) -> Field:
        """Create a field from a dictionary.

        Args:
            field_dict: A dictionary representation of the field.

        Returns:
            A field object created from the dictionary.

        """
        return cls(
            field_dict.get("x", 0),
            field_dict.get("y", 0),
            field_dict.get("vx", 0.0),
            field_dict.get("vy", 0.0),
        )
