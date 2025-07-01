"""Wavelength Module

This module defines the `Wavelength` and `WavelengthGroup` classes for
managing wavelengths in optical simulations. The `Wavelength` class represents
a single wavelength, allowing for its value to be defined in various units and
converted to microns for internal consistency. The `WavelengthGroup` class
manages collections of `Wavelength` objects, providing functionality to work
with multiple wavelengths simultaneously.

Kramer Harrison, 2024
"""

import optiland.backend as be


class Wavelength:
    """Represents a wavelength value with support for unit conversion.

    Args:
        value (float): The value of the wavelength.
        is_primary (bool): Indicates whether the wavelength is a primary
            wavelength.
        unit (str): The unit of the wavelength value. Defaults to 'microns'.

    Methods:
        _convert_to_um(): Converts the wavelength value to microns.

    """

    def __init__(self, value, is_primary=True, unit="um"):
        self._value = value
        self.is_primary = is_primary
        self._unit = unit.lower()
        self._value_in_um = self._convert_to_um()

    @property
    def value(self):
        """float: the value of the wavelength"""
        return self._value_in_um

    @property
    def unit(self):
        """str: the unit of the wavelength"""
        return "um"

    @unit.setter
    def unit(self, new_unit):
        """Sets the unit of the wavelength.

        Args:
            new_unit (str): The new unit to set for the wavelength.

        """
        self._unit = new_unit.lower()
        self._value_in_um = self._convert_to_um()

    def _convert_to_um(self):
        """Converts the wavelength value to micrometers (um) based on the
        current unit.

        Returns:
            float: The converted wavelength value in micrometers.

        Raises:
            ValueError: If the current unit is not supported for conversion to
                micrometers. Supported units: 'nm', 'um', 'mm', 'cm', 'm'.

        """
        unit_conversion = {"nm": 0.001, "um": 1, "mm": 1000, "cm": 10000, "m": 1000000}

        if self._unit in unit_conversion:
            conversion_factor = unit_conversion[self._unit]
            return self._value * conversion_factor
        raise ValueError("Unsupported unit for conversion to microns.")

    def to_dict(self):
        """Get a dictionary representation of the wavelength.

        Returns:
            dict: A dictionary representation of the wavelength.

        """
        return {"value": self._value, "is_primary": self.is_primary, "unit": self._unit}

    @classmethod
    def from_dict(cls, data):
        """Create a Wavelength instance from a dictionary representation.

        Args:
            data (dict): A dictionary containing the wavelength data.

        Returns:
            Wavelength: A new Wavelength instance created from the data.

        """
        required_keys = {"value", "is_primary", "unit"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        return cls(
            value=data["value"],
            is_primary=data["is_primary"],
            unit=data["unit"],
        )


class WavelengthGroup:
    """Represents a group of wavelengths.

    Attributes:
        wavelengths (list): A list of Wavelength objects.

    Methods:
        num_wavelengths(): Returns the number of wavelengths in the group.
        primary_index(): Returns the index of the primary wavelength.
        primary_wavelength(): Returns the primary wavelength.
        add_wavelength(value, is_primary=True, unit='um'): Adds a new
            wavelength to the group.
        get_wavelength(wavelength_number): Returns the value of a specific
            wavelength.
        get_wavelengths(): Returns a list of all the wavelength values in the
            group.

    """

    def __init__(self):
        self.wavelengths = []

    @property
    def num_wavelengths(self):
        """int: the number of wavelengths"""
        return len(self.wavelengths)

    @property
    def primary_index(self):
        """int: the index of the primary wavelength"""
        for index, wavelength in enumerate(self.wavelengths):
            if wavelength.is_primary:
                return index

    @property
    def primary_wavelength(self):
        """float: the primary wavelength"""
        return self.wavelengths[self.primary_index]

    def add_wavelength(self, value, is_primary=True, unit="um"):
        """Adds a new wavelength to the list of wavelengths.

        Args:
            value (float): The value of the wavelength.
            is_primary (bool, optional): Indicates if the wavelength is
                primary. Default is True.
            unit (str, optional): The unit of the wavelength. Default is 'um'.

        """
        if is_primary:
            for wavelength in self.wavelengths:
                wavelength.is_primary = False

        if self.num_wavelengths == 0:
            is_primary = True

        self.wavelengths.append(Wavelength(value, is_primary, unit))

    def get_wavelength(self, wavelength_number):
        """Get the value of a specific wavelength.

        Args:
            wavelength_number (int): The index of the desired wavelength.

        Returns:
            float: The value of the specified wavelength.

        """
        return self.wavelengths[wavelength_number].value

    def get_wavelengths(self):
        """Returns a list of wavelength values.

        Returns:
            list: A list of wavelength values.

        """
        return [wave.value for wave in self.wavelengths]

    def to_dict(self):
        """Get a dictionary representation of the wavelength group.

        Returns:
            dict: A dictionary representation of the wavelength group.

        """
        return {"wavelengths": [wave.to_dict() for wave in self.wavelengths]}

    @classmethod
    def from_dict(cls, data):
        """Create a WavelengthGroup instance from a dictionary representation.

        Args:
            data (dict): A dictionary containing the wavelength group data.

        Returns:
            WavelengthGroup: A new WavelengthGroup instance created from the
                data.

        """
        if "wavelengths" not in data:
            raise ValueError('Missing required key: "wavelengths"')

        new_group = cls()
        for wave_data in data["wavelengths"]:
            new_group.add_wavelength(**wave_data)

        return new_group


def add_wavelengths(
    wavelength_group,
    min_value,
    max_value,
    num_wavelengths,
    unit="um",
    *,
    sampling="chebyshev",
    scale="log",
):
    """Add new wavelengths corresponding to the geometrically-spaced Chebyshev nodes

    Args:
        min_value (float): Minimum wavelength value.
        max_value (float): Maximum wavelength value.
        num_wavelengths (int) : The number of wavelengths to be added.
            Has to be an odd integer.
        unit (str, optional): The unit of the wavelength. Default is 'um'.
        sampling (str, optional): The sampling algorithm used. Defaults to 'chebyshev'.
            Currently supported options are:
                'chebyshev' - chebyshev nodes of the first type
                'uniform' - uniformly spaced nodes across the specified range
        scale (str, optional): space in which the nodes are sampled. Defaults to 'log'.
            Currently supported options are:
                'log' - nodes are sampled in the logarithms of wavelength.
                'frequency' - nodes sampled in the frequency domain.
                'wavelength' - nodes sampled in the frequency domain. Not recommended.


    """
    if (
        not isinstance(num_wavelengths, int)
        or num_wavelengths % 2 == 0
        or num_wavelengths <= 0
    ):
        raise ValueError("num_wavelengths must be an odd positive integer")

    if min_value <= 0 or max_value <= 0:
        raise ValueError("min_value and max_value must be positive")

    scale = scale.lower()
    if scale in {"freq", "frequency"}:
        scale = "frequency"
    elif scale in {"wavelength"}:
        scale = "wavelength"
    elif scale in {"log", "logarithmic"}:
        scale = "log"
    else:
        raise ValueError(f"Unknown scale: {scale!r}")

    if scale == "frequency":
        power = -1.0
    elif scale == "wavelength":
        power = 1.0

    nodes = be.arange(1.0, num_wavelengths + 1.0)

    if sampling == "chebyshev":
        nodes = 0.5 * (1.0 - be.cos((2 * nodes - 1) * be.pi / (2 * num_wavelengths)))

    elif sampling == "uniform":
        nodes = (nodes - 0.5) / num_wavelengths

    if scale == "log":
        span = be.log2(max_value / min_value)
        for i, node in enumerate(nodes):
            is_primary = i == num_wavelengths // 2
            value = min_value * 2 ** (span * node)
            wavelength_group.wavelengths.append(Wavelength(value, is_primary, unit))
    else:
        min_value = min_value**power
        max_value = max_value**power
        span = max_value - min_value
        for i, node in enumerate(nodes):
            is_primary = i == num_wavelengths // 2
            value = min_value + (span * node)
            wavelength_group.wavelengths.append(
                Wavelength(value ** (1.0 / power), is_primary, unit)
            )
