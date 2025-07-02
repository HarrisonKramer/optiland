"""Fields Module

This module provides classes for representing the field of view of an optical
system.

Kramer Harrison, 2023
"""

import optiland.backend as be


class Field:
    """Represents a field with specific properties.

    Attributes:
        field_type (str): The type of the field.
        x (int): The x-coordinate of the field.
        y (int): The y-coordinate of the field.
        vx (float): The vignette factor in the x-direction.
        vy (float): The vignette factor in the y-direction.

    """

    def __init__(
        self,
        field_type,
        x=0,
        y=0,
        vignette_factor_x=0.0,
        vignette_factor_y=0.0,
    ):
        self.field_type = field_type
        self.x = x
        self.y = y
        self.vx = vignette_factor_x
        self.vy = vignette_factor_y

    def to_dict(self):
        """Convert the field to a dictionary.

        Returns:
            dict: A dictionary representation of the field.

        """
        return {
            "field_type": self.field_type,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
        }

    @classmethod
    def from_dict(cls, field_dict):
        """Create a field from a dictionary.

        Args:
            field_dict (dict): A dictionary representation of the field.

        Returns:
            Field: A field object created from the dictionary.

        """
        if "field_type" not in field_dict:
            raise ValueError("Missing required keys: field_type")

        return cls(
            field_dict["field_type"],
            field_dict.get("x", 0),
            field_dict.get("y", 0),
            field_dict.get("vx", 0.0),
            field_dict.get("vy", 0.0),
        )


class FieldGroup:
    """A class representing a group of fields.

    Attributes:
        fields (list): A list of fields in the group.
        telecentric (bool): Whether the system is telecentric in object space.

    Methods:
        get_vig_factor(Hx, Hy): Returns the vignetting factors for given Hx
            and Hy values.
        get_field_coords: Returns the normalized coordinates of the fields.
        add_field(field): Adds a field to the group.
        get_field(field_number): Returns the field at the specified index.

    """

    def __init__(self):
        self.fields = []
        self.telecentric = False

    @property
    def x_fields(self):
        """be.ndarray: x field values."""
        return be.array([field.x for field in self.fields])

    @property
    def y_fields(self):
        """be.ndarray: y field values."""
        return be.array([field.y for field in self.fields])

    @property
    def max_x_field(self):
        """float: Maximum field value in the x-direction."""
        return be.max(self.x_fields)

    @property
    def max_y_field(self):
        """float: Maximum field value in the y-direction."""
        return be.max(self.y_fields)

    @property
    def max_field(self):
        """float: Maximum radial field value."""
        return be.max(be.sqrt(self.x_fields**2 + self.y_fields**2))

    @property
    def num_fields(self):
        """int: number of fields in field group"""
        return len(self.fields)

    @property
    def vx(self):
        """be.ndarray: Vignetting factors in x for each field."""
        return be.array([field.vx for field in self.fields])

    @property
    def vy(self):
        """be.ndarray: Vignetting factors in y for each field."""
        return be.array([field.vy for field in self.fields])

    def get_vig_factor(self, Hx, Hy):
        """Calculates the vignetting factors for a given field position.

        Note that the vignetting factors are interpolated using the nearest
        neighbor method.

        Args
            Hx (float): The normalized x component of the field.
            Hy (float): The normalized y component of the field.

        Returns:
            vx_new (float): The interpolated x-component of the
                vignetting factor.
            vy_new (float): The interpolated y-component of the
                vignetting factor.

        """
        max_field = self.max_field
        if max_field == 0:
            x_fields = self.x_fields
            y_fields = self.y_fields
        else:
            x_fields = self.x_fields / max_field
            y_fields = self.y_fields / max_field
        fields = be.stack((x_fields, y_fields), axis=-1)
        v_data = be.stack((self.vx, self.vy), axis=-1)
        result = be.nearest_nd_interpolator(fields, v_data, Hx, Hy)
        vx_new = result[..., 0]
        vy_new = result[..., 1]
        return vx_new, vy_new

    def get_field_coords(self):
        """Returns the coordinates of the fields.

        If the maximum field size is 0, it returns a single coordinate (0, 0).
        Otherwise, it calculates the normalized coordinates for each field
        based on the maximum field size.

        Returns:
            list[tuple[float, float]]: A list of tuples, where each tuple
            contains the (normalized_x, normalized_y) coordinates of a field.

        """
        max_field = self.max_field
        if max_field == 0:
            return [(0, 0)]
        return [
            (float(x / max_field), float(y / max_field))
            for x, y in zip(self.x_fields, self.y_fields)
        ]

    def add_field(self, field):
        """Add a field to the list of fields.

        Args:
            field: The field to be added.

        """
        self.fields.append(field)

    def get_field(self, field_number):
        """Retrieve the field at the specified field_number.

        Args:
            field_number (int): The index of the field to retrieve.

        Returns:
            Field: The field at the specified index.

        Raises:
            IndexError: If the field_number is out of range.

        """
        return self.fields[field_number]

    def set_telecentric(self, is_telecentric):
        """Specify whether the system is telecentric in object space.

        Args:
            is_telecentric (bool): Whether the system is telecentric in object
                space.

        """
        self.telecentric = is_telecentric

    def to_dict(self):
        """Convert the field group to a dictionary.

        Returns:
            dict: A dictionary representation of the field group.

        """
        return {
            "fields": [field.to_dict() for field in self.fields],
            "telecentric": self.telecentric,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a field group from a dictionary.

        Args:
            data (dict): A dictionary representation of the field group.

        Returns:
            FieldGroup: A field group object created from the dictionary.

        """
        field_group = cls()
        for field_dict in data["fields"]:
            field_group.add_field(Field.from_dict(field_dict))
        field_group.set_telecentric(data["telecentric"])
        return field_group
