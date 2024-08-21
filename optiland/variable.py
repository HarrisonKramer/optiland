"""Optiland Distribution Module

This module provides a set of classes that represent different types of
variables within an optical system, such as radius, conic, and thickness of
optical surfaces. Each variable type is defined as a class that inherits from
the VariableBehavior base class, which provides a common interface for getting
and updating the value of the variable.

Kramer Harrison, 2024
"""
from abc import ABC, abstractmethod
import numpy as np


class VariableBehavior(ABC):
    """
    Represents the behavior of a variable in an optic system.

    Args:
        optic (Optic): The optic system to which the variable belongs.
        surface_number (int): The surface number of the variable.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic system to which the variable belongs.
        _surfaces (SurfaceGroup): The group of surfaces in the optic system.
        surface_number (int): The surface number of the variable.
    """

    def __init__(self, optic, surface_number, **kwargs):
        self.optic = optic
        self._surfaces = self.optic.surface_group
        self.surface_number = surface_number

    @abstractmethod
    def get_value(self):
        """
        Get the value of the variable.

        Returns:
            The value of the variable.
        """
        pass

    @abstractmethod
    def update_value(self, new_value):
        """
        Update the value of the variable.

        Args:
            new_value: The new value of the variable.
        """
        pass

    def scale(self, value):
        """
        Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale
        """
        pass

    def inverse_scale(self, scaled_value):
        """
        Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale
        """
        pass


class RadiusVariable(VariableBehavior):
    """
    Represents a variable for the radius of a surface in an optic.

    Args:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object that contains the surface.
        surface_number (int): The index of the surface in the optic.

    Methods:
        get_value(): Returns the current value of the radius.
        update_value(new_value): Updates the value of the radius.
    """

    def __init__(self, optic, surface_number, **kwargs):
        super().__init__(optic, surface_number, **kwargs)

    def get_value(self):
        """
        Returns the current value of the radius.

        Returns:
            float: The current value of the radius.
        """
        value = self._surfaces.radii[self.surface_number]
        return self.scale(value)

    def update_value(self, new_value):
        """
        Updates the value of the radius.

        Args:
            new_value (float): The new value of the radius.
        """
        new_value = self.inverse_scale(new_value)
        self.optic.set_radius(new_value, self.surface_number)

    def scale(self, value):
        """
        Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale
        """
        return value / 100.0 - 1.0

    def inverse_scale(self, scaled_value):
        """
        Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale
        """
        return (scaled_value + 1.0) * 100.0


class ConicVariable(VariableBehavior):
    """
    Represents a variable for the conic constant of a surface in an optic.

    Args:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The index of the surface in the optic.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The index of the surface in the optic.

    Methods:
        get_value: Returns the current conic constant of the surface.
        update_value: Updates the conic value of the surface.
    """

    def __init__(self, optic, surface_number, **kwargs):
        super().__init__(optic, surface_number, **kwargs)

    def get_value(self):
        """
        Returns the current conic constant of the surface.

        Returns:
            float: The conic constant of the surface.
        """
        return self._surfaces.conic[self.surface_number]

    def update_value(self, new_value):
        """
        Updates the conic value of the surface.

        Args:
            new_value (float): The new conic constant to set.

        """
        self.optic.set_conic(new_value, self.surface_number)

    def scale(self, value):
        """
        Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale
        """
        return value

    def inverse_scale(self, scaled_value):
        """
        Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale
        """
        return scaled_value


class ThicknessVariable(VariableBehavior):
    """
    Represents a variable for the thickness of an optic surface.

    Args:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.

    Methods:
        get_value(): Returns the current thickness value of the surface.
        update_value(new_value): Updates the thickness value of the surface.
    """

    def __init__(self, optic, surface_number, **kwargs):
        super().__init__(optic, surface_number, **kwargs)

    def get_value(self):
        """
        Returns the current thickness value of the surface.

        Returns:
            float: The current thickness value.
        """
        value = self._surfaces.get_thickness(self.surface_number)[0]
        return self.scale(value)

    def update_value(self, new_value):
        """
        Updates the thickness value of the surface.

        Args:
            new_value (float): The new thickness value.
        """
        new_value = self.inverse_scale(new_value)
        self.optic.set_thickness(new_value, self.surface_number)

    def scale(self, value):
        """
        Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale
        """
        return value / 10.0 - 1.0

    def inverse_scale(self, scaled_value):
        """
        Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale
        """
        return (scaled_value + 1.0) * 10.0


class IndexVariable(VariableBehavior):
    """
    Represents a variable for the index of refraction at a specific surface
    and wavelength.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The surface number where the variable is applied.
        wavelength (float): The wavelength at which the index of refraction is
            calculated.
        **kwargs: Additional keyword arguments.

    Attributes:
        optic (Optic): The optic object to which the surface belongs.
        surface_number (int): The number of the surface.
        wavelength (float): The wavelength at which the index of refraction is
            calculated.

    Methods:
        get_value(): Returns the value of the index of refraction at the
            specified surface and wavelength.
        update_value(new_value): Updates the value of the index of refraction
            at the specified surface.
    """

    def __init__(self, optic, surface_number, wavelength, **kwargs):
        super().__init__(optic, surface_number, **kwargs)
        self.wavelength = wavelength

    def get_value(self):
        """
        Returns the value of the index of refraction at the specified surface
        and wavelength.

        Returns:
            float: The value of the index of refraction.
        """
        n = self.optic.n(self.wavelength)
        value = n[self.surface_number]
        return self.scale(value)

    def update_value(self, new_value):
        """
        Updates the value of the index of refraction at the specified surface.

        Args:
            new_value (float): The new value of the index of refraction.
        """
        new_value = self.inverse_scale(new_value)
        self.optic.set_index(new_value, self.surface_number)

    def scale(self, value):
        """
        Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale
        """
        return value - 1.5

    def inverse_scale(self, scaled_value):
        """
        Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale
        """
        return scaled_value + 1.5


class AsphereCoeffVariable(VariableBehavior):
    """
    Represents a variable for an aspheric coefficient in an optical system.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The index of the surface in the optical system.
        coeff_number (int): The index of the aspheric coefficient.
        **kwargs: Additional keyword arguments.

    Attributes:
        coeff_number (int): The index of the aspheric coefficient.
    """

    def __init__(self, optic, surface_number, coeff_number, **kwargs):
        super().__init__(optic, surface_number, **kwargs)
        self.coeff_number = coeff_number

    def get_value(self):
        """
        Get the current value of the aspheric coefficient.

        Returns:
            float: The current value of the aspheric coefficient.
        """
        surf = self._surfaces.surfaces[self.surface_number]
        value = surf.geometry.c[self.coeff_number]
        return self.scale(value)

    def update_value(self, new_value):
        """
        Update the value of the aspheric coefficient.

        Args:
            new_value (float): The new value of the aspheric coefficient.
        """
        new_value = self.inverse_scale(new_value)
        self.optic.set_asphere_coeff(new_value, self.surface_number,
                                     self.coeff_number)

    def scale(self, value):
        """
        Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale
        """
        return value * 10 ** (4 + 2 * self.coeff_number)

    def inverse_scale(self, scaled_value):
        """
        Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale
        """
        return scaled_value / 10 ** (4 + 2 * self.coeff_number)


class PolynomialCoeffVariable(VariableBehavior):
    """
    Represents a variable for a polynomial coefficient of a PolynomialGeometry.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The index of the surface in the optical system.
        coeff_index (tuple(int, int)): The (x, y) indices of the polynomial
            coefficient.
        **kwargs: Additional keyword arguments.

    Attributes:
        coeff_number (int): The index of the polynomial coefficient.
    """

    def __init__(self, optic, surface_number, coeff_index, **kwargs):
        super().__init__(optic, surface_number, **kwargs)
        self.coeff_index = coeff_index

    def get_value(self):
        """
        Get the current value of the polynomial coefficient.

        Returns:
            float: The current value of the polynomial coefficient.
        """
        surf = self._surfaces.surfaces[self.surface_number]
        i, j = self.coeff_index
        try:
            value = surf.geometry.c[i][j]
        except IndexError:
            pad_width_i = max(0, i + 1 - surf.geometry.c.shape[0])
            pad_width_j = max(0, j + 1 - surf.geometry.c.shape[1])
            c_new = np.pad(surf.geometry.c,
                           pad_width=((0, pad_width_i), (0, pad_width_j)),
                           mode='constant',
                           constant_values=0)
            surf.geometry.c = c_new
            value = 0
        return self.scale(value)

    def update_value(self, new_value):
        """
        Update the value of the polynomial coefficient.

        Args:
            new_value (float): The new value of the polynomial coefficient.
        """
        new_value = self.inverse_scale(new_value)
        surf = self.optic.surface_group.surfaces[self.surface_number]
        i, j = self.coeff_index
        try:
            surf.geometry.c[i][j] = new_value
        except IndexError:
            pad_width_i = max(0, i + 1 - surf.geometry.c.shape[0])
            pad_width_j = max(0, j + 1 - surf.geometry.c.shape[1])
            c_new = np.pad(surf.geometry.c,
                           pad_width=((0, pad_width_i), (0, pad_width_j)),
                           mode='constant',
                           constant_values=0)
            c_new[i][j] = new_value
            surf.geometry.c = c_new

    def scale(self, value):
        """
        Scale the value of the variable for improved optimization performance.

        Args:
            value: The value to scale
        """
        return value

    def inverse_scale(self, scaled_value):
        """
        Inverse scale the value of the variable.

        Args:
            scaled_value: The scaled value to inverse scale
        """
        return scaled_value


class Variable:
    """
    Represents a variable in an optical system.

    Args:
        optic (OpticalSystem): The optical system to which the variable
            belongs.
        type (str): The type of the variable. Valid types are 'radius',
            'conic', 'thickness', 'index' and 'asphere_coeff'.
        **kwargs: Additional keyword arguments to be stored as attributes of
            the variable.

    Attributes:
        optic (OpticalSystem): The optical system to which the variable
            belongs.
        type_name (str): The type of the variable.
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

    def __init__(self, optic, type_name, min_val=None, max_val=None, **kwargs):
        self.optic = optic
        self.type = type_name
        self.min_val = min_val
        self.max_val = max_val
        self.kwargs = kwargs

        for key, value in kwargs.items():
            if key in self.allowed_attributes():
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a recognized attribute")

        self.variable = self._get_variable()

    @staticmethod
    def allowed_attributes():
        """
        This method returns a set of strings that are the names of allowed
        attributes.
        """
        return {'surface_number', 'coeff_number', 'wavelength', 'coeff_index'}

    def _get_variable(self):
        """
        Get the variable.

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
            'asphere_coeff': AsphereCoeffVariable,
            'polynomial_coeff': PolynomialCoeffVariable
        }

        variable_class = variable_types.get(self.type)

        # Instantiate the class if it exists
        if variable_class:
            return variable_class(**behavior_kwargs)
        else:
            raise ValueError(f'Invalid variable type "{self.type}"')

    @property
    def value(self):
        """Return the value of the variable.

        Returns:
            float: The value of the variable.

        Raises:
            ValueError: If the variable type is invalid.
        """
        return self.variable.get_value()

    @property
    def bounds(self):
        """Returns the bounds of the variable as a tuple.

        Returns:
            tuple: the bounds of the variable
        """
        min_val = (self.variable.scale(self.min_val)
                   if self.min_val is not None else None)
        max_val = (self.variable.scale(self.max_val)
                   if self.max_val is not None else None)
        return min_val, max_val

    def update(self, new_value):
        """Update variable to a new value.

        Args:
            new_value (float): The new value with which to update the variable.

        Raises:
            ValueError: If the variable type is invalid.
        """
        self.variable.update_value(new_value)
