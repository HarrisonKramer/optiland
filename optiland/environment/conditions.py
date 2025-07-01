"""Defines environmental conditions for optical calculations.

This module provides a dataclass to store and manage environmental
parameters relevant to optical modeling, such as temperature, pressure,
humidity, and CO2 concentration.
"""

from dataclasses import dataclass

@dataclass
class EnvironmentalConditions:
    """Stores environmental parameters for refractive index calculations.

    Attributes:
        pressure (float): Ambient pressure in Pascals (Pa).
            Defaults to 101325 Pa (standard atmosphere).
        temperature (float): Ambient temperature in degrees Celsius (°C).
            Defaults to 15 °C (standard temperature for Ciddor model).
        relative_humidity (float): Relative humidity as a fraction (0 to 1).
            Defaults to 0.0.
        co2_ppm (float): Carbon dioxide concentration in parts per million.
            Defaults to 400 ppm.
        wavelength (float, optional): Wavelength of light in microns (μm).
            This is optional and can be provided if the conditions are
            wavelength-specific for a particular calculation, though many
            models require it as a direct input. Defaults to None.
    """
    pressure: float = 101325.0  # Pascals
    temperature: float = 15.0  # Degrees Celsius
    relative_humidity: float = 0.0  # Fraction (0 to 1)
    co2_ppm: float = 400.0  # Parts per million
    wavelength: float = None # Microns
