"""Optiland Fields Module

This module provides classes for representing fields in an optical system

Kramer Harrison, 2023
"""
import numpy as np


class Field:
    """
    Represents a field with specific properties.

    Attributes:
        field_type (str): The type of the field.
        x (int): The x-coordinate of the field.
        y (int): The y-coordinate of the field.
        vx (float): The vignette factor in the x-direction.
        vy (float): The vignette factor in the y-direction.
    """

    def __init__(self, field_type, x=0, y=0,
                 vignette_factor_x=0.0, vignette_factor_y=0.0):
        self.field_type = field_type
        self.x = x
        self.y = y
        self.vx = vignette_factor_x
        self.vy = vignette_factor_y


class FieldGroup:
    """
    A class representing a group of fields.

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
        """np.array: x field values"""
        return np.array([field.x for field in self.fields])

    @property
    def y_fields(self):
        """np.array: y field values"""
        return np.array([field.y for field in self.fields])

    @property
    def max_x_field(self):
        """np.array: max field in x"""
        return np.max(self.x_fields)

    @property
    def max_y_field(self):
        """np.array: max field in y"""
        return np.max(self.y_fields)

    @property
    def max_field(self):
        """np.array: max field in radial coordinates"""
        return np.max(np.sqrt(self.x_fields**2 + self.y_fields**2))

    @property
    def num_fields(self):
        """int: number of fields in field group"""
        return len(self.fields)

    @property
    def vx(self):
        """np.array: vignetting factors in x"""
        return np.array([field.vx for field in self.fields])

    @property
    def vy(self):
        """np.array: vignetting factors in y"""
        return np.array([field.vy for field in self.fields])

    def get_vig_factor(self, Hx, Hy):
        """
        Calculates the vignetting factors for a given field position.

        Parameters:
            Hx (float): The normalized x component of the field.
            Hy (float): The normalized y component of the field.

        Returns:
            vx_new (float): The interpolated x-component of the
                vignetting factor.
            vy_new (float): The interpolated y-component of the
                vignetting factor.

        Raises:
            NotImplementedError: If the system is not rotationally-symmetric.
        """
        if np.all(self.x_fields == 0):  # assume rotationally symmetric
            idx_sorted = np.argsort(self.y_fields)
            if self.max_y_field == 0:
                h_sorted = np.zeros(self.num_fields)
            else:
                h_sorted = self.y_fields[idx_sorted] / self.max_y_field
            vx_sorted = self.vx[idx_sorted]
            vy_sorted = self.vy[idx_sorted]

            h = np.sqrt(Hx**2 + Hy**2)

            vx_new = np.interp(h, h_sorted, vx_sorted)
            vy_new = np.interp(h, h_sorted, vy_sorted)
            return vx_new, vy_new
        else:
            raise NotImplementedError('Currently only rotationally-symmetric '
                                      'systems may set vignetting factors')

    def get_field_coords(self):
        """
        Returns the coordinates of the fields.

        If the maximum field size is 0, it returns a single coordinate (0, 0).
        Otherwise, it calculates the normalized coordinates for each field
        based on the maximum field size.

        Returns:
            list: A list of tuples representing the coordinates of the fields.
        """
        max_field = self.max_field
        if max_field == 0:
            return [(0, 0)]
        return [(float(x/max_field), float(y/max_field))
                for x, y in zip(self.x_fields, self.y_fields)]

    def add_field(self, field):
        """
        Add a field to the list of fields.

        Parameters:
            field: The field to be added.
        """
        self.fields.append(field)

    def get_field(self, field_number):
        """
        Retrieve the field at the specified field_number.

        Args:
            field_number (int): The index of the field to retrieve.

        Returns:
            Field: The field at the specified index.

        Raises:
            IndexError: If the field_number is out of range.
        """
        return self.fields[field_number]

    def set_telecentric(self, is_telecentric):
        """
        Speocify whether the system is telecentric in object space.

        Args:
            is_telecentric (bool): Whether the system is telecentric in object
                space.
        """
        self.telecentric = is_telecentric
