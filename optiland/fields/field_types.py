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
        """Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.

        """
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
        """Calculate the position of the object in the paraxial optical system.

        Args:
            Hy (float): The normalized field height.
            y1 (ndarray): The initial y-coordinate of the ray.
            EPL (float): The entrance pupil location.

        Returns:
            tuple: A tuple containing the y and z coordinates of the object
                position.

        """
        field_y = optic.fields.max_field * Hy
        y = -be.tan(be.radians(field_y)) * EPL
        z = optic.surface_group.positions[1]
        y0 = y1 + y
        z0 = be.ones_like(y1) * z
        return y0, z0

    def scale_chief_ray_for_field(self, optic, y_obj_unit, u_obj_unit, y_img_unit):
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        This is used in the paraxial chief_ray calculation. It uses the results
        of a forward and backward "unit" trace from the stop to determine the
        final scaling factor.

        Args:
            optic (Optic): The optical system.
            y_obj_unit (float): The object-space height of the unit ray.
            u_obj_unit (float): The object-space angle of the unit ray.
            y_img_unit (float): The image-space height of the unit ray.

        Returns:
            float: The scaling factor.
        """
        max_field_angle = optic.fields.max_y_field
        target_slope = be.tan(be.deg2rad(max_field_angle))
        return target_slope / u_obj_unit

    def _get_starting_z_offset(self, optic):
        """Calculate the starting ray z-coordinate offset for systems with an
        object at infinity. This is relative to the first surface of the optic.

        This method chooses a starting point that is equivalent to the entrance
        pupil diameter of the optic.

        Args:
            optic (Optic): The optical system being traced.

        Returns:
            float: The z-coordinate offset relative to the first surface.

        """
        z = optic.surface_group.positions[1:-1]
        offset = optic.paraxial.EPD()
        return offset - be.min(z)


class ObjectHeightField(BaseFieldDefinition):
    """Defines fields by height on the object surface."""

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.

        Raises:
            ValueError: If the field type is "object_height" for an object at
                infinity.
        """
        self._validate_object_infinite(optic)
        obj = optic.object_surface
        max_field = optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        x0 = be.array(field_x)
        y0 = be.array(field_y)
        z0 = obj.geometry.sag(x0, y0) + obj.geometry.cs.z
        return x0, y0, z0

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """Calculate the position of the object in the paraxial optical system.

        Args:
            Hy (float): The normalized field height.
            y1 (ndarray): The initial y-coordinate of the ray.
            EPL (float): The entrance pupil location.

        Returns:
            tuple: A tuple containing the y and z coordinates of the object
                position.

        Raises:
            ValueError: If the field type is "object_height" and the object is
                at infinity.

        """
        self._validate_object_infinite(optic)
        obj = optic.object_surface
        field_y = optic.fields.max_field * Hy
        y = -field_y
        z = obj.geometry.cs.z
        y0 = be.ones_like(y1) * y
        z0 = be.ones_like(y1) * z
        return y0, z0

    def scale_chief_ray_for_field(self, optic, y_obj_unit, u_obj_unit, y_img_unit):
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        This is used in the paraxial chief_ray calculation. It uses the results
        of a forward and backward "unit" trace from the stop to determine the
        final scaling factor.

        Args:
            optic (Optic): The optical system.
            y_obj_unit (float): The object-space height of the unit ray.
            u_obj_unit (float): The object-space angle of the unit ray.
            y_img_unit (float): The image-space height of the unit ray.

        Returns:
            float: The scaling factor.
        """
        max_field_height = optic.fields.max_y_field
        return max_field_height / y_obj_unit

    def _validate_object_infinite(self, optic):
        """Check if the object surface is at infinity.

        Args:
            optic (Optic): The optical system being traced.

        Raises:
            ValueError: If the object surface is at infinity.
        """
        if optic.object_surface.is_infinite:
            raise ValueError("Object surface is at infinity.")


class ParaxialImageHeightField(BaseFieldDefinition):
    """Defines fields by the chief ray's paraxial height at the image plane."""

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.

        Raises:
            ValueError: If the field type is "object_height" for an object at
                infinity.
        """
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
        """Calculate the position of the object in the paraxial optical system.

        Args:
            Hy (float): The normalized field height.
            y1 (ndarray): The initial y-coordinate of the ray.
            EPL (float): The entrance pupil location.

        Returns:
            tuple: A tuple containing the y and z coordinates of the object
                position.

        """
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
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        This is used in the paraxial chief_ray calculation. It uses the results
        of a forward and backward "unit" trace from the stop to determine the
        final scaling factor.

        Args:
            optic (Optic): The optical system.
            y_obj_unit (float): The object-space height of the unit ray.
            u_obj_unit (float): The object-space angle of the unit ray.
            y_img_unit (float): The image-space height of the unit ray.

        Returns:
            float: The scaling factor.
        """
        max_image_height = optic.fields.max_y_field
        return max_image_height / y_img_unit

    def _get_starting_z_offset(self, optic):
        """Calculate the starting ray z-coordinate offset for systems with an
        object at infinity. This is relative to the first surface of the optic.

        This method chooses a starting point that is equivalent to the entrance
        pupil diameter of the optic.

        Args:
            optic (Optic): The optical system being traced.

        Returns:
            float: The z-coordinate offset relative to the first surface.

        """
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
