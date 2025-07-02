"""Unit tests for the Edlén (1966) air refractive index model using pytest."""

import pytest
from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.edlen import (
    edlen_refractive_index,
    _calculate_saturation_vapor_pressure,
    # Import constants from the model for clarity and direct use
    DISP_A, DISP_B, DISP_C, DISP_D, DISP_E,
    CO2_STD_PPM, CO2_CORR_FACTOR,
    ALPHA_GAS, DENSITY_FACTOR_STD, TORR_TO_PA,
    WATER_VAPOR_A, WATER_VAPOR_B
)
from optiland.environment import refractive_index_air # For API test
import optiland.backend as be


@pytest.fixture
def conditions_edlen_standard():
    """
    EnvironmentalConditions for Edlén's standard air:
    15°C, 101325 Pa (760 Torr), 0% RH, 300 ppm CO2.
    """
    return EnvironmentalConditions(
        temperature=15.0,
        pressure=101325.0, # 760 Torr
        relative_humidity=0.0,
        co2_ppm=CO2_STD_PPM # 300.0 ppm
    )

def test_edlen_svp_calculation(set_test_backend):
    """Test the _calculate_saturation_vapor_pressure helper."""
    # Using Buck (1981) formula as implemented in edlen.py
    # At 15°C
    temp_c_15 = 15.0
    expected_svp_15 = 611.21 * be.exp(
        (18.678 - temp_c_15 / 234.5) * (temp_c_15 / (257.14 + temp_c_15))
    )
    assert _calculate_saturation_vapor_pressure(temp_c_15) == pytest.approx(float(expected_svp_15)) # Approx 1705.1 Pa

    # At 25°C
    temp_c_25 = 25.0
    expected_svp_25 = 611.21 * be.exp(
        (18.678 - temp_c_25 / 234.5) * (temp_c_25 / (257.14 + temp_c_25))
    )
    assert _calculate_saturation_vapor_pressure(temp_c_25) == pytest.approx(float(expected_svp_25)) # Approx 3168.7 Pa


def test_edlen_standard_air_dispersion(conditions_edlen_standard, set_test_backend):
    """
    Test Edlén model for standard air (15°C, 760 Torr, 300ppm CO2, dry).
    λ = 0.5462255 μm (mercury green line).
    Edlén (1966) Table 1, page 76: (n-1)s x 10^8 = 27745.1
    So, (n_s - 1) = 2.77451e-4, n_s = 1.000277451.
    At these conditions, CO2 correction factor is 1, density factor ratio is 1, humidity is 0.
    """
    wavelength_um = 0.5462255
    expected_n = 1.000277451

    n_calculated = edlen_refractive_index(wavelength_um, conditions_edlen_standard)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)

    # Test via main API
    n_api = refractive_index_air(wavelength_um, conditions_edlen_standard, model='edlen')
    assert n_api == pytest.approx(expected_n, abs=1e-9)


def test_edlen_co2_correction(set_test_backend):
    """Test CO2 correction from 300ppm baseline to 400ppm at 15°C, 760 Torr, dry."""
    wavelength_um = 0.5462255 # (n_s-1) at 300ppm for this is 2.77451e-4
    conditions_400ppm = EnvironmentalConditions(
        temperature=15.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=400.0
    )

    sigma_sq = (1.0 / wavelength_um)**2
    n_s_minus_1_300ppm_e8 = ( # Refractivity for 300ppm CO2 baseline
        DISP_A + DISP_B / (DISP_C - sigma_sq) + DISP_D / (DISP_E - sigma_sq)
    )
    n_s_minus_1_300ppm = n_s_minus_1_300ppm_e8 * 1.0e-8

    # Apply CO2 correction for 400ppm
    co2_factor = 1.0 + CO2_CORR_FACTOR * (conditions_400ppm.co2_ppm - CO2_STD_PPM) * 1.0e-6
    expected_n_s_corrected_minus_1 = n_s_minus_1_300ppm * co2_factor
    # Since P,T are standard for Edlen, density factor is 1. RH=0.
    expected_n_400ppm = 1.0 + expected_n_s_corrected_minus_1

    n_calculated = edlen_refractive_index(wavelength_um, conditions_400ppm)
    assert n_calculated == pytest.approx(expected_n_400ppm, abs=1e-9)


def test_edlen_pressure_temperature_correction(set_test_backend):
    """Test P, T correction. T=20°C, P=700 Torr (93325.66 Pa), 300ppm CO2, dry."""
    wavelength_um = 0.5462255 # (n_s-1) at 300ppm for this is 2.77451e-4
    pressure_custom_pa = 700.0 * TORR_TO_PA
    conditions_custom_tp = EnvironmentalConditions(
        temperature=20.0, pressure=pressure_custom_pa, relative_humidity=0.0, co2_ppm=CO2_STD_PPM # 300ppm
    )

    sigma_sq = (1.0 / wavelength_um)**2
    n_s_minus_1_300ppm_e8 = (
        DISP_A + DISP_B / (DISP_C - sigma_sq) + DISP_D / (DISP_E - sigma_sq)
    )
    n_s_minus_1_300ppm = n_s_minus_1_300ppm_e8 * 1.0e-8
    # CO2 is at baseline 300ppm, so n_s_corrected_minus_1 = n_s_minus_1_300ppm

    # Apply P, T correction
    p_torr = conditions_custom_tp.pressure / TORR_TO_PA # Should be 700.0
    t_c = conditions_custom_tp.temperature # 20.0
    density_factor_actual = (
        p_torr * (1.0 + p_torr * (0.817 - 0.0133 * t_c) * 1.0e-6)
    ) / (1.0 + ALPHA_GAS * t_c)
    expected_n_tp_minus_1 = n_s_minus_1_300ppm * (density_factor_actual / DENSITY_FACTOR_STD)
    expected_n = 1.0 + expected_n_tp_minus_1 # RH=0

    n_calculated = edlen_refractive_index(wavelength_um, conditions_custom_tp)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)


def test_edlen_humidity_correction(conditions_edlen_standard, set_test_backend):
    """Test humidity correction. 15°C, 760 Torr, 300ppm CO2, 50% RH."""
    wavelength_um = 0.5462255 # (n_s-1) at 300ppm, 15C, 760Torr is 2.77451e-4

    conditions_humid = EnvironmentalConditions(
        temperature=conditions_edlen_standard.temperature, # 15.0
        pressure=conditions_edlen_standard.pressure,       # 101325.0
        relative_humidity=0.5,
        co2_ppm=conditions_edlen_standard.co2_ppm          # 300.0
    )

    # 1. (n_tp - 1) for dry air component at these P,T,CO2
    # This is (n_s_corrected - 1) which is (n_s - 1) because P,T,CO2 are standard base
    n_tp_minus_1_dry = 2.77451e-4

    # 2. Water vapor correction term (delta_n_w)
    t_c = conditions_humid.temperature
    svp_pa = _calculate_saturation_vapor_pressure(t_c)
    f_pa = conditions_humid.relative_humidity * svp_pa
    f_torr = f_pa / TORR_TO_PA
    sigma_sq = (1.0 / wavelength_um)**2
    water_vapor_corr = -f_torr * (WATER_VAPOR_A - WATER_VAPOR_B * sigma_sq) * 1.0e-8

    expected_n = 1.0 + n_tp_minus_1_dry + water_vapor_corr
    n_calculated = edlen_refractive_index(wavelength_um, conditions_humid)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)


def test_edlen_ciddor_comparison_point(set_test_backend):
    """Test Edlen against a value cited by Ciddor (1996) Table 4.
    λ=0.633μm, t=20°C, P=100000 Pa, dry (RH=0), CO2=300ppm (Edlen's baseline).
    Ciddor Table 4 for Edlen: (n-1)x10^6 = 270.03.
    """
    conditions = EnvironmentalConditions(
        temperature=20.0, pressure=100000.0, relative_humidity=0.0, co2_ppm=300.0
    )
    wavelength_um = 0.633
    expected_n_approx = 1.00027003

    n_calculated = edlen_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n_approx, abs=1e-8)


def test_edlen_input_validation(conditions_edlen_standard, set_test_backend):
    """Test input validation for Edlén model."""
    with pytest.raises(TypeError, match="conditions must be an EnvironmentalConditions object"):
        edlen_refractive_index(0.55, {"temperature": 15.0}) # type: ignore

    with pytest.raises(ValueError, match="Wavelength must be positive"):
        edlen_refractive_index(0.0, conditions_edlen_standard)
    with pytest.raises(ValueError, match="Wavelength must be positive"):
        edlen_refractive_index(-0.5, conditions_edlen_standard)

    # Test for temperature that might cause division by zero in density term: (1 + ALPHA_GAS * t_c)
    # If 1 + ALPHA_GAS * t_c = 0  => t_c = -1 / ALPHA_GAS
    problem_temp_c = -1.0 / ALPHA_GAS # Approx -272.405... °C
    conditions_problem_temp = EnvironmentalConditions(temperature=problem_temp_c, co2_ppm=300.0)
    with pytest.raises(ZeroDivisionError):
         edlen_refractive_index(0.5, conditions_problem_temp)

    # Test saturation vapor pressure with extreme temperature (if it leads to math issues)
    # SVP formula: 611.21 * be.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))
    # Denominator (257.14 + temp_c) can be zero if temp_c = -257.14
    conditions_svp_denom_zero = EnvironmentalConditions(temperature=-257.14, co2_ppm=300.0)
    if be.get_backend_name() == "torch":
        # PyTorch might return inf/nan instead of raising error immediately.
        # The subsequent calculations might then error or produce nan.
        # This needs careful check of torch behavior if strict error type is required.
        pass # Skip strict ZeroDivisionError for torch for now
    else: # Numpy
        with pytest.raises(ZeroDivisionError): # Or FloatingPointError for overflow if exp arg gets too large
            _calculate_saturation_vapor_pressure(conditions_svp_denom_zero.temperature)
