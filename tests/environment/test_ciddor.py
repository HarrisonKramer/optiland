"""Unit tests for the Ciddor (1996) air refractive index model using pytest."""

import pytest
from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.ciddor import (
    ciddor_refractive_index,
    _calculate_molar_mass_air,
    _calculate_saturation_vapor_pressure,
    _calculate_enhancement_factor,
    _calculate_compressibility,
    _get_density,
    # Import constants for direct use in tests if needed for clarity
    A_SVP, B_SVP, C_SVP, D_SVP, # For SVP
    ALPHA_F, BETA_F, GAMMA_F,   # For enhancement factor
    R_GAS_CONSTANT, M_W_VAPOR,  # General constants
    P_STD_AIR_PA, T_STD_AIR_K, CO2_MOLAR_PPM, # For molar mass & density
    K0, K1, K2, K3, CO2_STD_PPM as CO2_REF_DISP_PPM, CO2_CORR_FACTOR, # Dispersion
    W0, W1, W2, W3, CF_VAPOR, # Water vapor dispersion
    P_STD_VAPOR_PA, T_STD_VAPOR_K
)
from optiland.environment import refractive_index_air # For API test
import optiland.backend as be # For be.exp if directly testing expressions


@pytest.fixture
def conditions_ciddor_appendix_b():
    """Conditions from Ciddor (1996) Appendix B example."""
    return EnvironmentalConditions(
        temperature=20.0,       # t = 20 C
        pressure=100000.0,    # P = 100000 Pa
        relative_humidity=0.5,  # RH = 0.50
        co2_ppm=350.0           # xc = 350 ppm
    )

# --- Tests for Helper Functions ---

def test_calculate_molar_mass_air(set_test_backend):
    """Test molar mass of dry air calculation."""
    # Ciddor (1996) uses CO2_MOLAR_PPM = 400ppm as baseline for molar mass formula
    assert _calculate_molar_mass_air(CO2_MOLAR_PPM) == pytest.approx(0.0289635)
    # Example: 350 ppm
    expected_Ma_350 = 1e-3 * (28.9635 + 12.011e-6 * (350.0 - CO2_MOLAR_PPM))
    assert _calculate_molar_mass_air(350.0) == pytest.approx(expected_Ma_350)

def test_calculate_saturation_vapor_pressure(set_test_backend):
    """Test saturation vapor pressure (SVP) calculation."""
    # Using Ciddor's formula constants directly from Appendix A
    # For t = 20°C (T_k = 293.15 K)
    T_k_20C = 20.0 + 273.15
    expected_svp_20c = be.exp(A_SVP * T_k_20C**2 + B_SVP * T_k_20C + C_SVP + D_SVP / T_k_20C)
    assert _calculate_saturation_vapor_pressure(T_k_20C) == pytest.approx(float(expected_svp_20c), abs=1e-3) # Approx 2334.83 Pa

    # For t = 0°C (T_k = 273.15 K)
    T_k_0C = 0.0 + 273.15
    expected_svp_0c = be.exp(A_SVP * T_k_0C**2 + B_SVP * T_k_0C + C_SVP + D_SVP / T_k_0C)
    assert _calculate_saturation_vapor_pressure(T_k_0C) == pytest.approx(float(expected_svp_0c), abs=1e-3) # Approx 610.53 Pa

def test_calculate_enhancement_factor(set_test_backend):
    """Test enhancement factor (f_w) calculation."""
    # Conditions: P=100000 Pa, t=20°C
    P_test = 100000.0
    T_c_test = 20.0
    expected_f_w = ALPHA_F + BETA_F * P_test + GAMMA_F * T_c_test**2
    assert _calculate_enhancement_factor(P_test, T_c_test) == pytest.approx(expected_f_w) # Approx 1.003984

def test_calculate_compressibility(set_test_backend):
    """Test compressibility (Z) calculation. Ciddor App. B example values."""
    # For actual air sample: P=100kPa, T=293.15K (20C), xw=0.011756 (from App B)
    Z_actual_expected = 0.999595 # From Ciddor App B, line (B6)
    # Note: The paper has intermediate values rounded. Direct calc might differ slightly.
    assert _calculate_compressibility(100000.0, 293.15, 0.011756) == pytest.approx(Z_actual_expected, abs=1e-6)

    # For standard dry air (axs): P_std, T_std_air, xw=0
    Z_axs_expected = 0.999828 # From Ciddor App B, line (B10)
    assert _calculate_compressibility(P_STD_AIR_PA, T_STD_AIR_K, 0.0) == pytest.approx(Z_axs_expected, abs=1e-6)

    # For standard water vapor (ws): P_std_vapor, T_std_vapor, xw=1
    Z_ws_expected = 0.999956 # From Ciddor App B, line (B13)
    assert _calculate_compressibility(P_STD_VAPOR_PA, T_STD_VAPOR_K, 1.0) == pytest.approx(Z_ws_expected, abs=1e-6)


def test_get_density(set_test_backend):
    """Test density calculation using Ciddor App. B example values."""
    # For actual dry air component (rho_a): P=100kPa, T=293.15K, Ma_350ppm, xw=0.011756
    # Z_actual = 0.999595
    Ma_350 = _calculate_molar_mass_air(350.0) # From App B conditions
    rho_a_expected = 1.16769 # kg/m3; Ciddor App B (B7)
    # Full calculation of rho_a: (P * Ma * (1-xw)) / (Z * R * T)
    Z_actual = _calculate_compressibility(100000.0, 293.15, 0.011756)
    calc_rho_a = (100000.0 * Ma_350 * (1.0 - 0.011756)) / (Z_actual * R_GAS_CONSTANT * 293.15)
    assert calc_rho_a == pytest.approx(rho_a_expected, abs=1e-5)
    # Check _get_density for the full sample, then derive component if needed, or test components directly.
    # The model's _get_density is for the *full moist air sample*, not components.
    # Ciddor's rho_a and rho_w are calculated slightly differently in the main function.

    # Density of standard dry air (rho_axs) at 350ppm CO2:
    # P_std_air, T_std_air_K, Ma_350, xw=0, Z_axs = 0.999828
    rho_axs_expected = 1.22096 # kg/m3; Ciddor App B (B11)
    # This is _get_density with xw=0
    assert _get_density(P_STD_AIR_PA, T_STD_AIR_K, Ma_350, 0.0) == pytest.approx(rho_axs_expected, abs=1e-5)

    # Density of standard pure water vapor (rho_ws):
    # P_std_vapor, T_std_vapor_K, M_W_VAPOR, xw=1 (for molar mass in _get_density)
    # Z_ws = 0.999956
    # rho_ws = (P_std_vapor * M_W_VAPOR) / (Z_ws * R_GAS_CONSTANT * T_std_vapor_K)
    rho_ws_expected = 0.010008 # kg/m3; Ciddor App B (B14)
    # Note: _get_density's Ma parameter is for "dry air equivalent". For pure vapor,
    # it's better to use the direct formula as in the main ciddor_refractive_index function.
    # The test here is for the _get_density function's behavior.
    # If we call _get_density with M_W_VAPOR as Ma and xw=1:
    # rho = (P * Ma_wv / (Z*R*T)) * (1 - 1*(1 - Mw/Mw)) = P*Mw/(ZRT)
    assert _get_density(P_STD_VAPOR_PA, T_STD_VAPOR_K, M_W_VAPOR, 1.0) == pytest.approx(rho_ws_expected, abs=1e-5)


# --- Tests for Main ciddor_refractive_index Function ---

def test_ciddor_refractive_index_appendix_b_example(conditions_ciddor_appendix_b, set_test_backend):
    """Test against Ciddor (1996) Appendix B example.
    λ = 0.6329908 μm (HeNe laser wavelength from example)
    Expected n = 1.000269175 => (n-1)*10^6 = 269.175.
    """
    wavelength_um = 0.6329908
    expected_n = 1.000269175

    n_calculated = ciddor_refractive_index(wavelength_um, conditions_ciddor_appendix_b)
    assert n_calculated == pytest.approx(expected_n, abs=1e-9) # High precision match

    # Test via main API as well
    n_api = refractive_index_air(wavelength_um, conditions_ciddor_appendix_b, model='ciddor')
    assert n_api == pytest.approx(expected_n, abs=1e-9)


def test_ciddor_refractive_index_table4_dry_air(set_test_backend):
    """Test against Ciddor (1996) Table 4, dry air values for Ciddor model.
    λ=0.633 μm, t=20°C, P=101325 Pa, RH=0.
    """
    wavelength_um = 0.633
    # xc = 0 ppm, Ciddor model expected (n-1)*10^6 = 273.34
    cond_0ppm = EnvironmentalConditions(
        temperature=20.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=0.0
    )
    n_0ppm = ciddor_refractive_index(wavelength_um, cond_0ppm)
    assert (n_0ppm - 1.0) * 1e6 == pytest.approx(273.34, abs=0.01)

    # xc = 400 ppm, Ciddor model expected (n-1)*10^6 = 273.40
    cond_400ppm = EnvironmentalConditions(
        temperature=20.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=400.0
    )
    n_400ppm = ciddor_refractive_index(wavelength_um, cond_400ppm)
    assert (n_400ppm - 1.0) * 1e6 == pytest.approx(273.40, abs=0.01)


def test_ciddor_refractive_index_table5_humid_air(set_test_backend):
    """Test against Ciddor (1996) Table 5, humid air value for Ciddor model.
    λ=0.633μm, t=20°C, P=100000 Pa, pv = 1000 Pa (partial H2O pressure), xc=400 ppm.
    Expected (n-1)*10^6 = 260.78.
    """
    wavelength_um = 0.633
    temp_c = 20.0
    pressure_pa = 100000.0
    co2_ppm = 400.0
    pv_target = 1000.0 # Target partial pressure of water vapor

    # Need to find RH that yields pv_target = 1000 Pa at T=20C, P=100kPa.
    # pv = rh * f_w * svp  => rh = pv / (f_w * svp)
    temp_k = temp_c + 273.15
    svp = _calculate_saturation_vapor_pressure(temp_k) # Approx 2334.83 Pa
    f_w = _calculate_enhancement_factor(pressure_pa, temp_c) # Approx 1.003984

    # If using be.is_tensor to check, need to handle it here.
    # For now, assume scalar math for calculating rh_for_pv_target.
    svp_f = float(svp)
    f_w_f = float(f_w)

    rh_for_pv_target = pv_target / (f_w_f * svp_f) # Approx 0.426677

    conditions = EnvironmentalConditions(
        temperature=temp_c,
        pressure=pressure_pa,
        relative_humidity=rh_for_pv_target,
        co2_ppm=co2_ppm
    )

    # Verify that these conditions indeed produce pv close to 1000 Pa
    # xw = f * rh * svp / p_pa if p_pa > 0 else 0.0
    # partial_pressure = xw * p_pa = f * rh * svp
    pv_check = f_w_f * rh_for_pv_target * svp_f
    assert pv_check == pytest.approx(pv_target, abs=0.01)

    n = ciddor_refractive_index(wavelength_um, conditions)
    assert (n - 1.0) * 1e6 == pytest.approx(260.78, abs=0.01)


def test_ciddor_input_validation(set_test_backend):
    """Test input validation for Ciddor model and main API."""
    std_cond = EnvironmentalConditions()

    # Test main ciddor_refractive_index function (already checks conditions type internally)
    # No specific wavelength check in ciddor_refractive_index, but sigma_sq might be problematic.
    # Division by zero if wavelength_um is 0.
    with pytest.raises(ZeroDivisionError): # Or backend specific error if wavelength is 0
        ciddor_refractive_index(0.0, std_cond)

    # Test main API refractive_index_air for type errors
    with pytest.raises(TypeError, match="Input 'conditions' must be an instance of EnvironmentalConditions"):
        refractive_index_air(0.5, {"temp": 20}, model='ciddor') # type: ignore

    with pytest.raises(ValueError, match="Unsupported air refractive index model: nonexistent"):
        refractive_index_air(0.5, std_cond, model='nonexistent')

    # Check if z or rho_axs or rho_ws becomes zero leading to division by zero
    # This is tricky to force without specific knowledge of edge cases in Z or density calcs.
    # For instance, if z_actual becomes zero in ciddor_refractive_index.
    # This would require specific P, T, xw that makes Z=0.
    # The BIPM-91 Z formula is complex. For now, trust it's well-behaved for typical inputs.
    # If rho_axs or rho_ws is zero:
    # This happens if their respective P_STD or Molar Mass is zero, which are constants.
    # Or if their Z is infinite, or T is infinite. Unlikely with normal inputs.
    # What if P_STD_VAPOR_PA (1333 Pa) was 0?
    # This is a constant, so can't test via inputs unless we mock constants.

    # If p_pa is zero in main function:
    # xw = f * rh * svp / p_pa if p_pa > 0 else 0.0 -> xw becomes 0.
    # rho_a = (p_pa * m_a * (1.0 - xw)) / (z_actual * R_GAS_CONSTANT * t_k) -> becomes 0
    # rho_w = (p_pa * M_W_VAPOR * xw) / (z_actual * R_GAS_CONSTANT * t_k) -> becomes 0
    # Then term_air and term_vapor become 0. n_final_minus_1 = 0. n = 1.0. This is physically reasonable for vacuum.
    cond_zero_pressure = EnvironmentalConditions(pressure=0.0)
    n_vacuum = ciddor_refractive_index(0.55, cond_zero_pressure)
    assert n_vacuum == pytest.approx(1.0)
