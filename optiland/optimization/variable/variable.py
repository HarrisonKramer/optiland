"""Variable Module

This module contains the Variable class, which represents a variable in an
optical system. This is the core class for defining variables in the
optimization process within Optiland. In general, this class is used to define
any arbitrary variable that can be optimized in an optical system. The class
provides a common interface for all types of variables, such as radius, conic,
thickness, index, asphere coefficients, etc. The input parameter 'type' is used
to specify the type of the variable.

Kramer Harrison, 2024
"""

from optiland.optimization.variable.asphere_coeff import AsphereCoeffVariable
from optiland.optimization.variable.chebyshev_coeff import ChebyshevCoeffVariable
from optiland.optimization.variable.conic import ConicVariable
from optiland.optimization.variable.decenter import DecenterVariable
from optiland.optimization.variable.index import IndexVariable
from optiland.optimization.variable.polynomial_coeff import PolynomialCoeffVariable
from optiland.optimization.variable.radius import RadiusVariable
from optiland.optimization.variable.reciprocal_radius import ReciprocalRadiusVariable
from optiland.optimization.variable.thickness import ThicknessVariable
from optiland.optimization.variable.tilt import TiltVariable
from optiland.optimization.variable.zernike_coeff import ZernikeCoeffVariable


class Variable:
    """Represents a variable in an optical system.

    Args:
        optic (OpticalSystem): The optical system to which the variable
            belongs.
        type (str): The type of the variable.
        min_val (float or None): The minimum value allowed for the variable.
            Defaults to None.
        max_val (float or None): The maximum value allowed for the variable.
            Defaults to None.
        apply_scaling (bool): Whether to apply scaling to the variable.
            Defaults to True.
        **kwargs: Additional keyword arguments to be stored as attributes of
            the variable.

    Properties:
        value: The current value of the variable.
        bounds: The bounds of the variable.

    Methods:
        update(new_value): Updates the variable to a new value.

    Raises:
        ValueError: If an invalid variable type is provided.

    """

    def __init__(
        self,
        optic,
        type_name,
        min_val=None,
        max_val=None,
        apply_scaling=True,
        **kwargs,
    ):
        self.optic = optic
        self.type = type_name
        self.min_val = min_val
        self.max_val = max_val
        self.apply_scaling = apply_scaling
        self.kwargs = kwargs

        for key, value in kwargs.items():
            if key in self.allowed_attributes():
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a recognized attribute")

        self.variable = self._get_variable()
        self.initial_value = self.value

    @staticmethod
    def allowed_attributes():
        """This method returns a set of strings that are the names of allowed
        attributes.
        """
        return {"surface_number", "coeff_number", "wavelength", "coeff_index", "axis"}

    def _get_variable(self):
        """Get the variable.

        Returns:
            The behavior of the variable, or None if an error occurs.

        """
        behavior_kwargs = {
            "type_name": self.type,
            "optic": self.optic,
            "apply_scaling": self.apply_scaling,
            **self.kwargs,
        }

        variable_types = {
            "radius": RadiusVariable,
            "conic": ConicVariable,
            "thickness": ThicknessVariable,
            "tilt": TiltVariable,
            "decenter": DecenterVariable,
            "index": IndexVariable,
            "asphere_coeff": AsphereCoeffVariable,
            "polynomial_coeff": PolynomialCoeffVariable,
            "chebyshev_coeff": ChebyshevCoeffVariable,
            "zernike_coeff": ZernikeCoeffVariable,
            "reciprocal_radius": ReciprocalRadiusVariable,
        }

        variable_class = variable_types.get(self.type)

        # Instantiate the class if it exists
        if variable_class:
            return variable_class(**behavior_kwargs)
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
        min_val = (
            self.variable.scale(self.min_val) if self.min_val is not None else None
        )
        max_val = (
            self.variable.scale(self.max_val) if self.max_val is not None else None
        )
        return min_val, max_val

    def update(self, new_value):
        """Update variable to a new value.

        Args:
            new_value (float): The new value with which to update the variable.

        Raises:
            ValueError: If the variable type is invalid.

        """
        self.variable.update_value(new_value)

    def reset(self):
        """Reset the variable to its initial value."""
        self.update(self.initial_value)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return str(self.variable)
