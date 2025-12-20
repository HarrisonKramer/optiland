"""Base Field Definition Module

This module defines the abstract base class for field types in optical systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland import Optic
    from optiland._types import BEArray, ScalarOrArray


class BaseFieldDefinition(ABC):
    """Abstract base class for defining how fields map to ray properties."""

    @abstractmethod
    def get_ray_origins(
        self,
        optic: Optic,
        Hx: ScalarOrArray,
        Hy: ScalarOrArray,
        Px: ScalarOrArray,
        Py: ScalarOrArray,
        vx: ScalarOrArray,
        vy: ScalarOrArray,
    ) -> tuple[ScalarOrArray, ScalarOrArray, ScalarOrArray]:
        """Calculate the initial positions for rays originating at the object.

        Args:
            Hx: Normalized x field coordinate.
            Hy: Normalized y field coordinate.
            Px: x-coordinate of the pupil point.
            Py: y-coordinate of the pupil point.
            vx: Vignetting factor in the x-direction.
            vy: Vignetting factor in the y-direction.

        Returns:
            A tuple containing the x, y, and z coordinates of the
                object position.

        """
        pass  # pragma: no cover

    @abstractmethod
    def get_paraxial_object_position(
        self, optic: Optic, Hy: ScalarOrArray, y1: ScalarOrArray, EPL: ScalarOrArray
    ) -> tuple[BEArray, BEArray]:
        """Calculate the position of the object in the paraxial optical system.

        Args:
            Hy: The normalized field height.
            y1: The initial y-coordinate of the ray.
            EPL: The entrance pupil location.

        Returns:
            A tuple containing the y and z coordinates of the object
                position.

        """
        pass  # pragma: no cover

    @abstractmethod
    def scale_chief_ray_for_field(
        self,
        optic: Optic,
        y_obj_unit: ScalarOrArray,
        u_obj_unit: ScalarOrArray,
        y_img_unit: ScalarOrArray,
    ) -> ScalarOrArray:
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        This is used in the paraxial chief_ray calculation. It uses the results
        of a forward and backward "unit" trace from the stop to determine the
        final scaling factor.

        Args:
            optic: The optical system.
            y_obj_unit: The object-space height of the unit ray.
            u_obj_unit: The object-space angle of the unit ray.
            y_img_unit: The image-space height of the unit ray.

        Returns:
            The scaling factor.

        """
        pass  # pragma: no cover

    def to_dict(self) -> dict:
        """Convert the field definition to a dictionary.

        Returns:
            dict: A dictionary representation of the field definition.

        """
        return {"field_type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, field_def_dict: dict) -> BaseFieldDefinition:
        """Create a field definition from a dictionary.

        Args:
            field_def_dict (dict): A dictionary representation of the field
                definition.

        Returns:
            BaseFieldDefinition: A field definition object created from the
                dictionary.

        """
        if "field_type" not in field_def_dict:
            raise ValueError("Missing required keys: field_type")

        field_type = field_def_dict["field_type"]

        if field_type == "AngleField":
            from .angle import AngleField

            return AngleField()
        elif field_type == "ObjectHeightField":
            from .object_height import ObjectHeightField

            return ObjectHeightField()
        elif field_type == "ParaxialImageHeightField":
            from .paraxial_image_height import ParaxialImageHeightField

            return ParaxialImageHeightField()
        elif field_type == "RealImageHeightField":
            from .real_image_height import RealImageHeightField

            return RealImageHeightField()
        else:
            raise ValueError(f"Unknown field definition: {field_type}")
