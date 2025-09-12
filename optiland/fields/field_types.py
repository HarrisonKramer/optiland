"""Field Types Module

This module defines different field types for optical systems, including angle,
object height, and paraxial image height fields. Each field type implements
methods to calculate ray origins and paraxial object positions based on the
field definition.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from optiland import Optic


class BaseFieldDefinition(ABC):
    """Abstract base class for defining how fields map to ray properties."""

    @abstractmethod
    def get_ray_origins(
        self,
        optic: Optic,
        Hx: be.ndarray | float,
        Hy: be.ndarray | float,
        Px: be.ndarray | float,
        Py: be.ndarray | float,
        vx: be.ndarray | float,
        vy: be.ndarray | float,
    ) -> tuple[be.ndarray, be.ndarray, be.ndarray]:
        """Calculates the initial positions (x0, y0, z0) for real rays."""
        pass

    @abstractmethod
    def get_paraxial_object_position(
        self, optic: Optic, Hy: float, y1: be.ndarray, EPL: float
    ) -> tuple[be.ndarray, be.ndarray]:
        """Calculates the object position (y0, z0) for a paraxial ray fan."""
        pass

    @abstractmethod
    def scale_chief_ray_for_field(
        self, optic: Optic, y_obj_unit: float, u_obj_unit: float, y_img_unit: float
    ) -> float:
        """Calculates scaling factor for a chief ray based on the field definition."""
        pass

    def to_dict(self) -> dict:
        """Convert the field definition to a dictionary.

        Returns:
            dict: A dictionary representation of the field definition.

        """
        return {"field_type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, field_def_dict: dict) -> "BaseFieldDefinition":
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
            return AngleField()
        elif field_type == "ObjectHeightField":
            return ObjectHeightField()
        elif field_type == "ParaxialImageHeightField":
            return ParaxialImageHeightField()
        else:
            raise ValueError(f"Unknown field definition: {field_type}")


class AngleField(BaseFieldDefinition):
    """Defines fields by angle (in degrees) relative to the optical axis."""

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculates the initial positions for rays originating at the object."""
        obj = optic.object_surface
        EPL = optic.paraxial.EPL()
        max_field = optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy

        if obj.is_infinite:
            EPD = optic.paraxial.EPD()
            offset = self._get_starting_z_offset(optic)
            x = -be.tan(be.radians(field_x)) * (offset + EPL)
            y = -be.tan(be.radians(field_y)) * (offset + EPL)
            z = optic.surface_group.positions[1] - offset
            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            z0 = be.full_like(Px, z)
        else:
            z0 = optic.surface_group.positions[0]
            x0 = -be.tan(be.radians(field_x)) * (EPL - z0)
            y0 = -be.tan(be.radians(field_y)) * (EPL - z0)
            if be.size(x0) == 1:
                x0 = be.full_like(Px, x0)
            if be.size(y0) == 1:
                y0 = be.full_like(Px, y0)
            if be.size(z0) == 1:
                z0 = be.full_like(Px, z0)
        return x0, y0, z0

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """Calculates the object position for a paraxial ray fan."""
        field_y = optic.fields.max_field * Hy
        y = -be.tan(be.radians(field_y)) * EPL
        z = optic.surface_group.positions[1]
        y0 = y1 + y
        z0 = be.ones_like(y1) * z
        return y0, z0

    def scale_chief_ray_for_field(self, optic, y_obj_unit, u_obj_unit, y_img_unit):
        """Calculates scaling factor for a chief ray based on the field definition."""
        max_field_angle = optic.fields.max_y_field
        target_slope = be.tan(be.deg2rad(max_field_angle))
        return target_slope / u_obj_unit

    def _get_starting_z_offset(self, optic):
        z = optic.surface_group.positions[1:-1]
        offset = optic.paraxial.EPD()
        return offset - be.min(z)


class ObjectHeightField(BaseFieldDefinition):
    """Defines fields by height on the object surface."""

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculates the initial positions for rays originating at the object."""
        obj = optic.object_surface
        if obj.is_infinite:
            raise ValueError(
                'Field type "object_height" is not supported for an object at infinity.'
            )
        max_field = optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        x0 = be.array(field_x)
        y0 = be.array(field_y)
        z0 = obj.geometry.sag(x0, y0) + obj.geometry.cs.z
        return x0, y0, z0

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """Calculates the object position for a paraxial ray fan."""
        obj = optic.object_surface
        if obj.is_infinite:
            raise ValueError(
                'Field type "object_height" is not supported for an object at infinity.'
            )
        field_y = optic.fields.max_field * Hy
        y = -field_y
        z = obj.geometry.cs.z
        y0 = be.ones_like(y1) * y
        z0 = be.ones_like(y1) * z
        return y0, z0

    def scale_chief_ray_for_field(self, optic, y_obj_unit, u_obj_unit, y_img_unit):
        """Calculates scaling factor for a chief ray based on the field definition."""
        max_field_height = optic.fields.max_y_field
        return max_field_height / y_obj_unit


class ParaxialImageHeightField(BaseFieldDefinition):
    """Defines fields by the chief ray's paraxial height at the image plane."""

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculates the initial positions for rays originating at the object."""
        y_img_target = optic.fields.max_field * Hy
        x_img_target = optic.fields.max_field * Hx

        y_img_unit, _ = self._trace_unit_chief_ray(optic, plane="image")
        y_obj_unit, u_obj_unit = self._trace_unit_chief_ray(optic, plane="object")

        if optic.object_surface.is_infinite:
            u_obj_y = u_obj_unit * (y_img_target / y_img_unit)
            u_obj_x = u_obj_unit * (x_img_target / y_img_unit)

            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()
            offset = self._get_starting_z_offset(optic)

            x = -u_obj_x * (offset + EPL)
            y = -u_obj_y * (offset + EPL)
            z = optic.surface_group.positions[1] - offset
            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            z0 = be.full_like(Px, z)
        else:
            y_obj = y_obj_unit * (y_img_target / y_img_unit)
            x_obj = y_obj_unit * (x_img_target / y_img_unit)
            x0 = x_obj
            y0 = y_obj
            z0 = (
                optic.object_surface.geometry.sag(x0, y0)
                + optic.object_surface.geometry.cs.z
            )
            if be.size(x0) == 1:
                x0 = be.full_like(Px, x0)
            if be.size(y0) == 1:
                y0 = be.full_like(Px, y0)
            if be.size(z0) == 1:
                z0 = be.full_like(Px, z0)
        return x0, y0, z0

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """Calculates the object position for a paraxial ray fan."""
        y_img_target = optic.fields.max_field * Hy
        y_img_unit, _ = self._trace_unit_chief_ray(optic, plane="image")
        y_obj_unit, u_obj_unit = self._trace_unit_chief_ray(optic, plane="object")

        if optic.object_surface.is_infinite:
            u_obj = u_obj_unit * (y_img_target / y_img_unit)
            y = u_obj * -EPL
            z = optic.surface_group.positions[1]
            y0 = y1 + y
            z0 = be.ones_like(y1) * z
        else:
            y_obj = y_obj_unit * (y_img_target / y_img_unit)
            y = y_obj
            z = optic.object_surface.geometry.cs.z
            y0 = be.ones_like(y1) * y
            z0 = be.ones_like(y1) * z
        return y0, z0

    def scale_chief_ray_for_field(self, optic, y_obj_unit, u_obj_unit, y_img_unit):
        """Calculates scaling factor for a chief ray based on the field definition."""
        max_image_height = optic.fields.max_y_field
        return max_image_height / y_img_unit

    def _get_starting_z_offset(self, optic):
        z = optic.surface_group.positions[1:-1]
        offset = optic.paraxial.EPD()
        return offset - be.min(z)

    def _trace_unit_chief_ray(self, optic, plane="image"):
        stop_idx = optic.surface_group.stop_index
        num_surf = optic.surface_group.num_surfaces
        pos = optic.surface_group.positions
        wavelength = optic.primary_wavelength

        if plane == "image":
            z_start = pos[stop_idx]
            y, u = optic.paraxial._trace_generic(
                y=0, u=1, z=z_start, wavelength=wavelength, skip=stop_idx
            )
            return y[-1], u[-1]
        elif plane == "object":
            z_start = pos[-1] - pos[stop_idx]
            skip = num_surf - stop_idx
            y, u = optic.paraxial._trace_generic(
                y=0, u=1, z=z_start, wavelength=wavelength, reverse=True, skip=skip
            )
            return y[-1], u[-1]
