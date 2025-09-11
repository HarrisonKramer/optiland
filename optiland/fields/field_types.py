"""Field Types Module

This module defines different field types for optical systems, including angle
and object height fields. Each field type implements methods to calculate ray
origins and paraxial object positions based on the field definition.

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
        self, optic: Optic, y_back_obj: float, u_back_obj: float
    ) -> float:
        """
        Calculates scaling factor for a unit chief ray based on the field definition.

        This is used in the paraxial chief_ray calculation. It determines whether
        to scale based on the resulting object height or angle of a back-traced ray.
        """
        pass

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
        # TODO: Update to use subclass registry
        if "field_type" not in field_def_dict:
            raise ValueError("Missing required keys: field_type")

        field_type = field_def_dict["field_type"]

        if field_type == "AngleField":
            return AngleField()
        elif field_type == "ObjectHeightField":
            return ObjectHeightField()
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

        Raises:
            ValueError: If the field type is "object_height" for an object at
                infinity.
        """
        obj = optic.object_surface
        EPL = optic.paraxial.EPL()
        max_field = optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy

        if obj.is_infinite:
            EPD = optic.paraxial.EPD()

            offset = self._get_starting_z_offset(optic)

            # x, y, z positions of ray starting points
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
            EPL (float): The effective focal length of the lens.

        Returns:
            tuple: A tuple containing the y and z coordinates of the object
                position.

        Raises:
            ValueError: If the field type is "object_height" and the object is
                at infinity.

        """
        field_y = optic.fields.max_field * Hy

        y = -be.tan(be.radians(field_y)) * EPL
        z = optic.surface_group.positions[1]

        y0 = y1 + y
        z0 = be.ones_like(y1) * z

        return y0, z0

    def scale_chief_ray_for_field(self, optic, y_back_obj, u_back_obj):
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        This is used in the paraxial chief_ray calculation. It determines whether
        to scale based on the resulting object height or angle of a back-traced ray.

        Args:
            optic (Optic): The optical system being traced.
            y_back_obj (float): The y-coordinate of the back-traced ray at the
                object surface.
            u_back_obj (float): The angle (in radians) of the back-traced ray
                at the object surface.

        Returns:
            float: The scaling factor for the chief ray.
        """
        max_field = optic.fields.max_y_field
        return be.tan(be.deg2rad(max_field)) / u_back_obj

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
        """Calculate the position of the object in the paraxial optical system.

        Args:
            Hy (float): The normalized field height.
            y1 (ndarray): The initial y-coordinate of the ray.
            EPL (float): The effective focal length of the lens.

        Returns:
            tuple: A tuple containing the y and z coordinates of the object
                position.

        Raises:
            ValueError: If the field type is "object_height" and the object is
                at infinity.

        """
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

    def scale_chief_ray_for_field(self, optic, y_back_obj, u_back_obj):
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        This is used in the paraxial chief_ray calculation. It determines whether
        to scale based on the resulting object height or angle of a back-traced ray.

        Args:
            optic (Optic): The optical system being traced.
            y_back_obj (float): The y-coordinate of the back-traced ray at the
                object surface.
            u_back_obj (float): The angle (in radians) of the back-traced ray
                at the object surface.

        Returns:
            float: The scaling factor for the chief ray.
        """
        max_field = optic.fields.max_y_field
        return max_field / y_back_obj


class ParaxialImageHeightField(BaseFieldDefinition):
    """Defines fields by the chief ray's paraxial height at the image plane."""

    def _get_chief_ray_start_for_field(
        self, optic: Optic, Hy: float
    ) -> tuple[float, float]:
        """
        Calculates the starting (y, u) in object space for a chief ray
        that will have a specific height on the image plane.
        """
        # 1. Trace a unit ray backwards from the stop center.
        stop_index = optic.surface_group.stop_index
        pos = optic.surface_group.positions
        z_start_back = pos[-1] - pos[stop_index]
        wavelength = optic.primary_wavelength
        num_surf = optic.surface_group.num_surfaces
        skip_back = num_surf - stop_index

        y_back, u_back = optic.paraxial._trace_generic(
            y=0,
            u=0.1,
            z=z_start_back,
            wavelength=wavelength,
            reverse=True,
            skip=skip_back,
        )

        y_img_unit = y_back[0]  # Height at the image plane for the unit ray

        # 2. Determine the scaling factor.
        desired_y_image = optic.fields.max_field * Hy
        if abs(y_img_unit) < 1e-9:
            raise RuntimeError(
                "Cannot use image height for an image-space telecentric system."
            )

        scale = desired_y_image / y_img_unit

        # 3. Calculate the initial ray parameters in object space.
        y_obj_start = -y_back[-1] * scale  # y is inverted for forward trace
        u_obj_start = u_back[-1] * scale

        return y_obj_start, u_obj_start

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        # For real rays, we need the chief ray's origin on the object surface.
        y0_chief, u0_chief = self._get_chief_ray_start_for_field(optic, Hy)
        # Note: This implementation assumes fields are only in Y. A more robust
        # version would handle Hx as well.

        obj = optic.object_surface
        if obj.is_infinite:
            raise NotImplementedError(
                "Paraxial Image Height is not yet supported for objects at infinity."
            )

        y0 = be.full_like(Py, y0_chief)
        x0 = be.zeros_like(Px)  # Assuming x-field is zero
        z0 = obj.geometry.sag(x0, y0) + obj.geometry.cs.z

        return x0, y0, z0

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        y0_chief, _ = self._get_chief_ray_start_for_field(optic, Hy)
        z0 = optic.object_surface.geometry.cs.z

        # The object height for the entire fan is defined by the chief ray's height.
        return be.full_like(y1, y0_chief), be.full_like(y1, z0)

    def scale_chief_ray_for_field(self, optic, y_back_obj, u_back_obj):
        # This is conceptually different. The field *is* the result of the trace.
        # We want the chief ray that produces y_image = max_field.
        # This requires a forward trace, which complicates the current chief_ray method.
        # For now, an exception is clearest.
        raise NotImplementedError(
            "`chief_ray` calculation is not directly applicable "
            "when field is defined by image height."
        )
