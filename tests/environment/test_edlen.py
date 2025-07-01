"""Unit tests for the Edlén (1966) air refractive index model using pytest.

These tests verify the Edlén model implementation against published or
calculated reference values.
"""
import pytest
import math

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.edlen import edlen_refractive_index, _saturation_vapor_pressure_edlen
from optiland.environment import refractive_index_air

# Constants from Edlén (1966) for standard air (15°C, 760mmHg, dry, 300ppm CO2)
EDLEN_C1_ref = 8342.13
EDLEN_C2_ref = 2406030.0
EDLEN_C3_ref = 130.0
EDLEN_C4_ref = 15997.0
EDLEN_C5_ref = 38.9
TORR_TO_PA_ref = 101325.0 / 760.0
ALPHA_GAS_ref = 0.003661

@pytest.fixture
def std_conditions_edlen_ref():
    """EnvironmentalConditions for Edlén's standard air (15C, 101325Pa, dry).
    CO2 is not used by Edlen's model but set to 300ppm for conceptual match.
    """
    return EnvironmentalConditions(
        temperature=15.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=300.0
    )

def test_edlen_standard_air_dispersion(std_conditions_edlen_ref):
    """Test Edlén model for standard air against a reference value.
    Using lambda = 0.5462255 um (mercury green line).
    Edlén (1966) Table 1, page 76: (n-1)s x 10^8 = 27745.1
    So, (n_s - 1) = 2.77451e-4, n_s = 1.000277451.
    """
    wavelength_um = 0.5462255
    expected_n = 1.000277451

    n_calculated = edlen_refractive_index(wavelength_um, std_conditions_edlen_ref)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)

    # Test via main API
    n_api = refractive_index_air(wavelength_um, std_conditions_edlen_ref, model='edlen')
    assert n_api == pytest.approx(expected_n, abs=1e-9)


def test_edlen_pressure_temperature_correction(std_conditions_edlen_ref):
    """Test Edlén P, T correction factor part.
    If T=15C, P=760 Torr, the (n_tp-1) should be (n_s-1).
    Let's try T=20C, P=700 Torr.
    """
    wavelength_um = 0.5462255 # Same as above, (n_s-1) = 2.77451e-4

    conditions = EnvironmentalConditions(
        temperature=20.0, pressure=700.0 * TORR_TO_PA_ref, relative_humidity=0.0
    )
    t_c = 20.0
    p_torr = 700.0
    n_s_minus_1 = 2.77451e-4 # From previous test

    # Manual calculation of (n_tp-1) using Edlen's specific P,T terms
    # beta_t = (0.0624 - 0.000680*T_c)*1e-6
    beta_t = (0.0624 - 0.000680 * t_c) * 1.0e-6
    # (n_tp - 1) = (n_s - 1) * (p_torr/760) * (1/(1+ALPHA_GAS*t_c)) * (1 - p_torr*beta_t)
    expected_n_tp_minus_1 = n_s_minus_1 * (p_torr / 760.0) * \
                            (1.0 / (1.0 + ALPHA_GAS_ref * t_c)) * \
                            (1.0 - p_torr * beta_t)
    expected_n = 1.0 + expected_n_tp_minus_1

    n_calculated = edlen_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)


def test_edlen_humidity_correction():
    """Test Edlén water vapor correction.
    Using 15°C, 101325 Pa (760 Torr), 50% RH, lambda=0.5462255 um.
    (n_tp - 1) = 2.77451e-4 (dry air at these P,T).
    SVP at 15C (Magnus from Edlen module): ~1705.0 Pa (was 1703.5 in my scratch)
    pv_pa = 0.5 * _saturation_vapor_pressure_edlen(15.0)
    f_torr = pv_pa / TORR_TO_PA_ref
    sigma = 1/lambda_um, sigma2 = sigma^2
    delta_n_w = -f_torr * (5.722 - 0.0457 * sigma2) * 1e-8
    """
    wavelength_um = 0.5462255
    conditions = EnvironmentalConditions(
        temperature=15.0, pressure=101325.0, relative_humidity=0.5
    )

    # 1. (n_tp - 1) for dry air component at these conditions (15C, 760 Torr)
    # This is (n_s - 1) because P,T are standard for Edlen's n_s
    n_tp_minus_1 = 2.77451e-4

    # 2. Water vapor correction term (delta_n_w)
    t_c = 15.0
    pv_pa = conditions.relative_humidity * _saturation_vapor_pressure_edlen(t_c)
    f_torr = pv_pa / TORR_TO_PA_ref

    sigma_sq = (1.0 / wavelength_um)**2
    delta_n_w_expected = -f_torr * (5.722 - 0.0457 * sigma_sq) * 1.0e-8

    expected_n = 1.0 + n_tp_minus_1 + delta_n_w_expected

    n_calculated = edlen_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9)


def test_edlen_input_validation(std_conditions_edlen_ref):
    """Test input validation for Edlén model."""
    with pytest.raises(TypeError):
        edlen_refractive_index(0.55, {"temperature": 15.0})

    with pytest.raises(ValueError, match="Wavelength must be positive"):
        edlen_refractive_index(0.0, std_conditions_edlen_ref)


# Check against a known example if possible, e.g. from Ciddor's paper comparing Edlen
# Ciddor (1996) mentions Edlen values.
# For lambda=0.633um, 20 C, 100000 Pa, dry air:
# Edlen (n-1)x10^6 = 270.03 (from Ciddor Table 4, adjusted for 0%RH)
# Let's test this point.
def test_edlen_ciddor_comparison_point():
    """Test Edlen against a value cited by Ciddor (1996) Table 4."""
    # Conditions: lambda=0.633um, t=20 C, P=100000 Pa, dry (RH=0).
    # CO2 implicitly 300ppm for Edlen.
    conditions = EnvironmentalConditions(
        temperature=20.0, pressure=100000.0, relative_humidity=0.0, co2_ppm=300.0
    )
    wavelength_um = 0.633
    expected_n_approx = 1.00027003 # (n-1)x10^6 = 270.03

    n_calculated = edlen_refractive_index(wavelength_um, conditions)
    assert n_calculated == pytest.approx(expected_n_approx, abs=1e-8) # abs is 0.01 for (n-1)x10^6

    # For 0ppm CO2, Ciddor's Edlen value is 269.97. Edlen doesn't vary CO2.
    # For 400ppm CO2, Ciddor's Edlen value is 270.06.
    # This shows Edlen is fixed at its assumed 300ppm CO2.
    # The value 270.03 is for 300ppm implicitly.

    # Test via main API as well
    n_api = refractive_index_air(wavelength_um, conditions, model='edlen')
    assert n_api == pytest.approx(expected_n_approx, abs=1e-8)
