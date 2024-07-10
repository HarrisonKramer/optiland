"""Optiland Distribution Module

This module provides a set of classes that represent different types of
variables within an optical system, such as radius, conic, and thickness of
optical surfaces. Each variable type is defined as a class that inherits from
the VariableBehavior base class, which provides a common interface for getting
and updating the value of the variable.

Kramer Harrison, 2024
"""


class VariableBehavior:
    def __init__(self, optic, surface_number, **kwargs):
        self.optic = optic
        self._surfaces = self.optic.surface_group
        self.surface_number = surface_number

    def get_value(self):
        raise NotImplementedError

    def update_value(self, new_value):
        raise NotImplementedError


class RadiusVariable(VariableBehavior):
    def __init__(self, optic, surface_number, **kwargs):
        super().__init__(optic, surface_number, **kwargs)

    def get_value(self):
        return self._surfaces.radii[self.surface_number]

    def update_value(self, new_value):
        self.optic.set_radius(new_value, self.surface_number)


class ConicVariable(VariableBehavior):
    def __init__(self, optic, surface_number, **kwargs):
        super().__init__(optic, surface_number, **kwargs)

    def get_value(self):
        return self._surfaces.conic[self.surface_number]

    def update_value(self, new_value):
        self.optic.set_conic(new_value, self.surface_number)


class ThicknessVariable(VariableBehavior):
    def __init__(self, optic, surface_number, **kwargs):
        super().__init__(optic, surface_number, **kwargs)

    def get_value(self):
        return self._surfaces.get_thickness(self.surface_number)[0]

    def update_value(self, new_value):
        self.optic.set_thickness(new_value, self.surface_number)


class IndexVariable(VariableBehavior):
    def __init__(self, optic, surface_number, wavelength, **kwargs):
        super().__init__(optic, surface_number, **kwargs)
        self.wavelength = wavelength

    def get_value(self):
        n = self.optic.n(self.wavelength)
        return n[self.surface_number]

    def update_value(self, new_value):
        self.optic.set_index(new_value, self.surface_number)


class AsphereCoeffVariable(VariableBehavior):
    def __init__(self, optic, surface_number, coeff_number, **kwargs):
        super().__init__(optic, surface_number, **kwargs)
        self.coeff_number = coeff_number

    def get_value(self):
        surf = self._surfaces.surfaces[self.surface_number]
        return surf.geometry.c[self.coeff_number]

    def update_value(self, new_value):
        self.optic.set_asphere_coeff(new_value, self.surface_number,
                                     self.coeff_number)


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

    def __init__(self, optic, type, min_val=None, max_val=None, **kwargs):
        self.optic = optic
        self.type = type
        self.min_val = min_val
        self.max_val = max_val
        self.kwargs = kwargs

        for key, value in kwargs.items():
            if key in self.allowed_attributes():
                setattr(self, key, value)
            else:
                # Handle unexpected attributes or raise a warning/error
                print(f"Warning: {key} is not a recognized attribute")

        self.variable_behavior = self._get_variable_behavior()

    @staticmethod
    def allowed_attributes():
        """
        Define allowed additional attributes here.
        This method returns a set of strings that are the names of allowed
        attributes.
        """
        return {'surface_number', 'coeff_number', 'wavelength'}

    def _get_variable_behavior(self):
        """
        Get the behavior of the variable.

        Returns:
            The behavior of the variable, or None if an error occurs.
        """
        behavior_kwargs = {
            'type_name': self.type,
            'optic': self.optic,
            **self.kwargs
        }

        variable_types = {
            'radius': RadiusVariable,
            'conic': ConicVariable,
            'thickness': ThicknessVariable,
            'index': IndexVariable,
            'asphere_coeff': AsphereCoeffVariable
        }

        variable_class = variable_types.get(self.type)

        # Instantiate the class if it exists
        if variable_class:
            return variable_class(**behavior_kwargs)
        else:
            return None

    @property
    def value(self):
        """Return the value of the variable.

        Returns:
            float: The value of the variable.

        Raises:
            ValueError: If the variable type is invalid.
        """
        if self.variable_behavior:
            return self.variable_behavior.get_value()
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
        if self.variable_behavior:
            self.variable_behavior.update_value(new_value)
        else:
            raise ValueError(f'Invalid variable type "{self.type}"')
