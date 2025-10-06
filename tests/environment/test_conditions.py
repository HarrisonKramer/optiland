# tests/environment/test_conditions.py
"""
Unit tests for the EnvironmentalConditions dataclass.
"""
from __future__ import annotations

import pytest

from optiland.environment.conditions import EnvironmentalConditions


def test_environmental_conditions_defaults():
    """
    Tests that EnvironmentalConditions initializes with the correct default
    values when no arguments are provided.
    """
    conditions = EnvironmentalConditions()
    assert conditions.pressure == 101325.0
    assert conditions.temperature == 15.0
    assert conditions.relative_humidity == 0.0
    assert conditions.co2_ppm == 400.0
    assert conditions.wavelength is None


def test_environmental_conditions_custom_values():
    """
    Tests that EnvironmentalConditions correctly stores custom values for all
    attributes provided during initialization.
    """
    conditions = EnvironmentalConditions(
        pressure=100000.0,
        temperature=20.0,
        relative_humidity=0.5,
        co2_ppm=350.0,
        wavelength=0.55,
    )
    assert conditions.pressure == 100000.0
    assert conditions.temperature == 20.0
    assert conditions.relative_humidity == 0.5
    assert conditions.co2_ppm == 350.0
    assert conditions.wavelength == 0.55


def test_environmental_conditions_partial_custom_values():
    """
    Tests that initializing with only some custom values correctly assigns
    them, while others remain as their defaults.
    """
    conditions = EnvironmentalConditions(temperature=25.0, co2_ppm=500.0)
    assert conditions.pressure == 101325.0  # Default
    assert conditions.temperature == 25.0  # Custom
    assert conditions.relative_humidity == 0.0  # Default
    assert conditions.co2_ppm == 500.0  # Custom
    assert conditions.wavelength is None  # Default


@pytest.mark.parametrize(
    "pressure, temperature, rh, co2, wl",
    [
        (101325, 15, 0.0, 400, None),
        (90000, 0, 0.25, 300, 0.633),
        (110000, 30, 1.0, 800, 1.55),
    ],
)
def test_environmental_conditions_parametrization(pressure, temperature, rh, co2, wl):
    """
    Tests EnvironmentalConditions with various valid parameter sets to ensure
    robustness.
    """
    conditions = EnvironmentalConditions(
        pressure=pressure,
        temperature=temperature,
        relative_humidity=rh,
        co2_ppm=co2,
        wavelength=wl,
    )
    assert conditions.pressure == pressure
    assert conditions.temperature == temperature
    assert conditions.relative_humidity == rh
    assert conditions.co2_ppm == co2
    assert conditions.wavelength == wl


def test_environmental_conditions_types():
    """
    Tests that attributes maintain their types (float or int) as provided,
    and that the `wavelength` attribute can be None.
    """
    # Test with float values
    conditions_float = EnvironmentalConditions(
        pressure=100000.0,
        temperature=20.0,
        relative_humidity=0.5,
        co2_ppm=350.0,
        wavelength=0.55,
    )
    assert isinstance(conditions_float.pressure, float)
    assert isinstance(conditions_float.temperature, float)
    assert isinstance(conditions_float.relative_humidity, float)
    assert isinstance(conditions_float.co2_ppm, float)
    assert isinstance(conditions_float.wavelength, float)

    # Test with integer values
    conditions_int = EnvironmentalConditions(
        pressure=100000, temperature=20, relative_humidity=0, co2_ppm=350
    )
    assert isinstance(conditions_int.pressure, int)
    assert isinstance(conditions_int.temperature, int)
    assert isinstance(conditions_int.relative_humidity, int)
    assert isinstance(conditions_int.co2_ppm, int)

    # Test with None for wavelength
    conditions_none_wl = EnvironmentalConditions()
    assert conditions_none_wl.wavelength is None


def test_environmental_conditions_mutability():
    """
    Tests that attributes of an EnvironmentalConditions instance can be
    mutated after instantiation.
    """
    conditions = EnvironmentalConditions()
    conditions.temperature = 25.0
    conditions.co2_ppm = 600.0
    assert conditions.temperature == 25.0
    assert conditions.co2_ppm == 600.0
    # Ensure other attributes remain unchanged
    assert conditions.pressure == 101325.0
    assert conditions.relative_humidity == 0.0