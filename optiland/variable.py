"""Optiland Distribution Module

This module defines the `Variable` class, which represents a variable
parameter within an optical system. These variables can include properties
such as radius, conic constant, thickness, and refractive index of surfaces
within the system. The `Variable` class provides methods to get and set the
values of these parameters, as well as to update them within the context of
the optical system's overall configuration.

Kramer Harrison, 2024
"""


class Variable:
    """
    Represents a variable in an optical system.

    Args:
        optic (OpticalSystem): The optical system to which the variable
            belongs.
        type (str): The type of the variable. Valid types are 'radius',
            'conic', 'thickness', and 'index'.
        **kwargs: Additional keyword arguments to be stored as attributes of
            the variable.

    Attributes:
        optic (OpticalSystem): The optical system to which the variable
            belongs.
        type (str): The type of the variable.
        min_val (float or None): The minimum value allowed for the variable.
            Defaults to None.
        max_val (float or None): The maximum value allowed for the variable.
            Defaults to None.

    Properties:
        value: The current value of the variable.
        bounds: The bounds of the variable.

    Methods:
        update(new_value): Updates the variable to a new value.

    Raises:
        ValueError: If an invalid variable type is provided.
    """

    def __init__(self, optic, type, **kwargs):
        self.__dict__.update(kwargs)
        self.optic = optic
        self.type = type

        self._surfaces = self.optic.surface_group

        if not hasattr(self, 'min_val'):
            self.min_val = None

        if not hasattr(self, 'max_val'):
            self.max_val = None

    @property
    def value(self):
        """Return the value of the variable.

        Returns:
            float: The value of the variable.

        Raises:
            ValueError: If the variable type is invalid.
        """
        if self.type == 'radius':
            return self._surfaces.radii[self.surface_number]
        elif self.type == 'conic':
            return self._surfaces.conic[self.surface_number]
        elif self.type == 'thickness':
            return self._surfaces.get_thickness(self.surface_number)[0]
        elif self.type == 'index':
            n = self.optic.n(self.wavelength)
            return n[self.surface_number]
        else:
            raise ValueError(f'Invalid variable type "{self.type}"')

    @property
    def bounds(self):
        """Returns the bounds of the variable as a tuple.

        Returns:
            tuple: the bounds of the variable
        """
        return (self.min_val, self.max_val)

    def update(self, new_value):
        """Update variable to a new value.

        Args:
            new_value (float): The new value with which to update the variable.

        Raises:
            ValueError: If the variable type is invalid.
        """
        if self.type == 'radius':
            self.optic.set_radius(new_value, self.surface_number)
        elif self.type == 'conic':
            self.optic.set_conic(new_value, self.surface_number)
        elif self.type == 'thickness':
            self.optic.set_thickness(new_value, self.surface_number)
        elif self.type == 'index':
            self.optic.set_thickness(new_value, self.surface_number)
        else:
            raise ValueError(f'Invalid variable type "{self.type}"')
