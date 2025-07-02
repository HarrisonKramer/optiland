"""Unit tests for the Kohlrausch air refractive index model using pytest."""

import pytest
from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.kohlrausch import (
    kohlrausch_refractive_index,
    # Import constants from the model for clarity and direct use
    DISP_A, DISP_B, DISP_C, DISP_D, DISP_E,
    T_REF_C, P_STD_PA, ALPHA_T
)
from optiland.environment import refractive_index_air # For API test
# import optiland.backend as be # Not strictly needed yet as Kohlrausch model is pure arithmetic

@pytest.fixture
def conditions_kohlrausch_ref():
    """
    EnvironmentalConditions for Kohlrausch's reference temperature (15°C)
    and standard pressure (101325 Pa). Humidity and CO2 are ignored by this model.
    """
    return EnvironmentalConditions(
        temperature=T_REF_C,    # 15.0 °C
        pressure=P_STD_PA,      # 101325.0 Pa
        relative_humidity=0.0,  # Ignored by model
        co2_ppm=400.0           # Ignored by model
    )

def test_kohlrausch_dispersion_at_ref_tp(conditions_kohlrausch_ref, set_test_backend):
    """
    Test Kohlrausch model at its reference T (15°C) and standard P (101325 Pa).
    At these P, T, the scaling factor for P/T should be 1.
    So, n-1 should be exactly the (n_ref - 1) from the dispersion formula part.
    Using λ = 0.55 μm.
    """
    wavelength_um = 0.550
    sigma_sq = (1.0 / wavelength_um)**2

    # Expected (n_ref - 1) from the dispersion formula
    expected_n_ref_minus_1_e8 = (
        DISP_A + DISP_B / (DISP_C - sigma_sq) + DISP_D / (DISP_E - sigma_sq)
    )
    expected_n_ref_minus_1 = expected_n_ref_minus_1_e8 * 1.0e-8
    # At T_REF_C and P_STD_PA, the scaling factor is 1.
    expected_n = 1.0 + expected_n_ref_minus_1

    n_calculated = kohlrausch_refractive_index(wavelength_um, conditions_kohlrausch_ref)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)

    # Test via main API
    n_api = refractive_index_air(wavelength_um, conditions_kohlrausch_ref, model='kohlrausch')
    assert n_api == pytest.approx(expected_n, abs=1e-9)

def test_kohlrausch_temperature_scaling(set_test_backend):
    """Test Kohlrausch model with temperature differing from T_REF_C (15°C). P is P_STD_PA."""
    wavelength_um = 0.550
    custom_temp_c = 25.0 # Different from T_REF_C
    conditions = EnvironmentalConditions(temperature=custom_temp_c, pressure=P_STD_PA)

    sigma_sq = (1.0 / wavelength_um)**2
    n_ref_minus_1_e8 = (
        DISP_A + DISP_B / (DISP_C - sigma_sq) + DISP_D / (DISP_E - sigma_sq)
    )
    n_ref_minus_1 = n_ref_minus_1_e8 * 1.0e-8

    # Apply T scaling. P scaling is 1 as pressure=P_STD_PA.
    relative_pressure = conditions.pressure / P_STD_PA # This is 1.0
    temp_scaling_denom = 1.0 + (conditions.temperature - T_REF_C) * ALPHA_T

    expected_n_minus_1 = (n_ref_minus_1 * relative_pressure) / temp_scaling_denom
    expected_n = 1.0 + expected_n_minus_1

    n_calculated = kohlrausch_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)

def test_kohlrausch_pressure_scaling(set_test_backend):
    """Test Kohlrausch model with pressure differing from P_STD_PA. T is T_REF_C."""
    wavelength_um = 0.550
    custom_pressure_pa = 90000.0 # Different from P_STD_PA
    conditions = EnvironmentalConditions(temperature=T_REF_C, pressure=custom_pressure_pa)

    sigma_sq = (1.0 / wavelength_um)**2
    n_ref_minus_1_e8 = (
        DISP_A + DISP_B / (DISP_C - sigma_sq) + DISP_D / (DISP_E - sigma_sq)
    )
    n_ref_minus_1 = n_ref_minus_1_e8 * 1.0e-8

    # Apply P scaling. T scaling is 1 as temperature=T_REF_C.
    relative_pressure = conditions.pressure / P_STD_PA
    temp_scaling_denom = 1.0 + (conditions.temperature - T_REF_C) * ALPHA_T # This is 1.0

    expected_n_minus_1 = (n_ref_minus_1 * relative_pressure) / temp_scaling_denom
    expected_n = 1.0 + expected_n_minus_1

    n_calculated = kohlrausch_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)


def test_kohlrausch_input_validation(conditions_kohlrausch_ref, set_test_backend):
    """Test input validation for Kohlrausch model."""
    with pytest.raises(TypeError, match="conditions must be an EnvironmentalConditions object"):
        kohlrausch_refractive_index(0.55, {"temperature": 15.0}) # type: ignore

    # Wavelength checks (from model's sigma_sq calculation)
    with pytest.raises(ValueError, match="Wavelength must be non-zero."):
        kohlrausch_refractive_index(0.0, conditions_kohlrausch_ref)
    # Negative wavelength also effectively leads to ValueError from sigma_sq, or non-physical result
    # The model itself doesn't check for wl > 0, but 1/wl**2 is problematic for wl<=0.
    # Let's assume non-zero is the primary check from ZeroDivisionError.

    # Temperature checks (from model's temp_scaling_denom)
    # temp_scaling_denom = 1.0 + (t_c - T_REF_C) * ALPHA_T
    # If temp_scaling_denom <= 0, raises ValueError
    # (t_c - T_REF_C) * ALPHA_T <= -1.0
    # t_c - T_REF_C <= -1.0 / ALPHA_T
    # t_c <= T_REF_C - (1.0 / ALPHA_T)
    critical_temp_offset = -1.0 / ALPHA_T # Approx -287.48 deg C from T_REF_C
    problem_temp_c = T_REF_C + critical_temp_offset

    conditions_bad_temp = EnvironmentalConditions(temperature=problem_temp_c)
    with pytest.raises(ValueError, match="non-positive denominator"):
        kohlrausch_refractive_index(0.55, conditions_bad_temp)

    conditions_even_worse_temp = EnvironmentalConditions(temperature=problem_temp_c - 10.0)
    with pytest.raises(ValueError, match="non-positive denominator"):
        kohlrausch_refractive_index(0.55, conditions_even_worse_temp)


@pytest.mark.parametrize(
    "temp_c, pressure_pa, wavelength_um, expected_n_approx",
    [
        # λ=0.5893 (Sodium D-line)
        (15.0, 101325.0, 0.5893, 1.00027570), # Ref T, Std P
        (0.0,  101325.0, 0.5893, 1.00029042), # Cooler T
        (25.0, 90000.0,  0.5893, 1.00023830), # Warmer T, lower P
        # λ=0.4861 (Hydrogen F-line, blue)
        (15.0, 101325.0, 0.4861, 1.00027914), # Ref T, Std P
    ]
)
def test_kohlrausch_various_conditions(temp_c, pressure_pa, wavelength_um, expected_n_approx, set_test_backend):
    """Test Kohlrausch with a few representative conditions and wavelengths."""
    conditions = EnvironmentalConditions(temperature=temp_c, pressure=pressure_pa)

    n_calculated = kohlrausch_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n_approx, abs=1e-8)

```
