"""Unit tests for the Kohlrausch air refractive index model using pytest.

These tests verify the implementation of the simplified Kohlrausch model
for various environmental conditions.
"""
import pytest
import math

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.kohlrausch import kohlrausch_refractive_index
from optiland.environment import refractive_index_air # For API test

# Constants used in the kohlrausch.py implementation for reference calculation
A_K_ref = 287.5
B_K_ref = 5.0
T0_KELVIN_ref = 273.15
P0_PASCAL_ref = 101325.0

@pytest.fixture
def default_conditions():
    """Pytest fixture for default EnvironmentalConditions."""
    return EnvironmentalConditions()

def test_kohlrausch_reference_conditions():
    """Test Kohlrausch model at its reference conditions (0°C, 101325 Pa)."""
    conditions = EnvironmentalConditions(temperature=0.0, pressure=101325.0)
    wavelength_um = 0.55

    # Expected n0-1 at reference: (A_K + B_K / lambda^2) * 1e-6
    expected_n0_minus_1 = (A_K_ref + B_K_ref / wavelength_um**2) * 1.0e-6
    expected_n = 1.0 + expected_n0_minus_1

    n_calculated = kohlrausch_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)

    # Test via main API
    n_api = refractive_index_air(wavelength_um, conditions, model='kohlrausch')
    assert n_api == pytest.approx(expected_n, abs=1e-9)


def test_kohlrausch_temperature_scaling():
    """Test Kohlrausch model with temperature differing from reference."""
    conditions = EnvironmentalConditions(temperature=20.0, pressure=101325.0) # 20°C
    wavelength_um = 0.55

    n0_minus_1_ref = (A_K_ref + B_K_ref / wavelength_um**2) * 1.0e-6

    t_k_actual = conditions.temperature + 273.15
    expected_n_minus_1 = n0_minus_1_ref * \
                         (conditions.pressure / P0_PASCAL_ref) * \
                         (T0_KELVIN_ref / t_k_actual)
    expected_n = 1.0 + expected_n_minus_1

    n_calculated = kohlrausch_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)


def test_kohlrausch_pressure_scaling():
    """Test Kohlrausch model with pressure differing from reference."""
    conditions = EnvironmentalConditions(temperature=0.0, pressure=90000.0) # Custom P
    wavelength_um = 0.55

    n0_minus_1_ref = (A_K_ref + B_K_ref / wavelength_um**2) * 1.0e-6

    t_k_actual = conditions.temperature + 273.15
    expected_n_minus_1 = n0_minus_1_ref * \
                         (conditions.pressure / P0_PASCAL_ref) * \
                         (T0_KELVIN_ref / t_k_actual)
    expected_n = 1.0 + expected_n_minus_1

    n_calculated = kohlrausch_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)


def test_kohlrausch_input_validation(default_conditions):
    """Test input validation for Kohlrausch model."""
    with pytest.raises(TypeError):
        kohlrausch_refractive_index(0.55, {"temperature": 0.0})

    with pytest.raises(ValueError, match="Wavelength must be positive"):
        kohlrausch_refractive_index(0.0, default_conditions)

    with pytest.raises(ValueError, match="Wavelength must be positive"):
        kohlrausch_refractive_index(-0.55, default_conditions)

    bad_temp_conditions = EnvironmentalConditions(temperature=-300.0) # Below abs zero
    with pytest.raises(ValueError, match="Absolute temperature must be positive"):
        kohlrausch_refractive_index(0.55, bad_temp_conditions)

# Example of how parameters might be used if the model was more complex
@pytest.mark.parametrize(
    "temp, pressure, wavelength, expected_n_approx",
    [
        (0.0, 101325.0, 0.5893, 1.0002996), # (287.5 + 5.0/0.5893^2)*1e-6 + 1
        (15.0, 101325.0, 0.6328, 1.0002743), # Scaled from its 0C ref
    ]
)
def test_kohlrausch_various_conditions(temp, pressure, wavelength, expected_n_approx):
    """Test Kohlrausch with a few representative conditions."""
    conditions = EnvironmentalConditions(temperature=temp, pressure=pressure)

    # Manual calculation for these specific cases based on the model's formula
    n0_minus_1_ref_case = (A_K_ref + B_K_ref / wavelength**2) * 1.0e-6
    t_k_actual_case = temp + 273.15

    if t_k_actual_case <=0: # Should not happen with test data
        pytest.skip("Temperature results in non-positive Kelvin.")

    calculated_n_minus_1 = n0_minus_1_ref_case * \
                           (pressure / P0_PASCAL_ref) * \
                           (T0_KELVIN_ref / t_k_actual_case)
    calculated_n = 1.0 + calculated_n_minus_1

    # The expected_n_approx in parametrize should match this `calculated_n`
    # For this test, we re-calculate the expected value based on the model logic
    # to ensure self-consistency and demonstrate parameterization.
    # The `expected_n_approx` in the decorator are more like external check values.

    n_model = kohlrausch_refractive_index(wavelength, conditions)
    assert n_model == pytest.approx(calculated_n, abs=1e-7)
    # Additionally, check against the pre-calculated expected_n_approx from parametrize
    assert n_model == pytest.approx(expected_n_approx, abs=1e-7)
