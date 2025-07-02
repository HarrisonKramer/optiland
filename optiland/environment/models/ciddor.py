"""Ciddor (1996) Air Refractive Index Model

This module provides functions to calculate the phase refractive index of air
based on the Ciddor (1996) formulation. This model is considered highly
accurate over a wide range of atmospheric conditions and wavelengths.

The method calculates the refractivity of moist air by determining the density
of the dry air and water vapor components under actual conditions and scaling
their standard refractivities accordingly. This approach relies on the BIPM-91
equation for the density of moist air.

References:
    Ciddor, P. E. (1996). Refractive index of air: new equations for the
    visible and near infrared. Applied Optics, 35(9), 1566-1573.

Kramer Harrison, 2025
"""

import numpy as np

from ..conditions import EnvironmentalConditions

# --- Model Constants from Ciddor (1996), Appendix A ---

# Universal Gas Constant [cite: 77]
R_GAS_CONSTANT = 8.314510  # J mol⁻¹ K⁻¹

# Molar mass of pure water vapor [cite: 77]
M_W_VAPOR = 0.018015  # kg mol⁻¹

# --- Standard Conditions ---
T_STD_AIR_C = 15.0  # Standard temperature for air, °C [cite: 37]
T_STD_AIR_K = 288.15  # Standard temperature for air, K
P_STD_AIR_PA = 101325.0  # Standard pressure for air, Pa [cite: 37]

T_STD_VAPOR_C = 20.0  # Standard temperature for water vapor, °C [cite: 59]
T_STD_VAPOR_K = 293.15  # Standard temperature for water vapor, K
P_STD_VAPOR_PA = 1333.0  # Standard pressure for water vapor, Pa [cite: 59]

CO2_STD_PPM = 450.0  # Standard CO₂ for refractivity equations, ppm [cite: 37]
CO2_MOLAR_PPM = 400.0  # Standard CO₂ for molar mass calc, ppm [cite: 77]

# --- Dispersion Constants ---
# For dry air with 450 ppm CO₂ at 15 °C, 101325 Pa [cite: 56, 206]
K0 = 238.0185  # μm⁻²
K1 = 5792105.0
K2 = 57.362  # μm⁻²
K3 = 167917.0
CO2_CORR_FACTOR = 0.534e-6  # ppm⁻¹

# For pure water vapor at 20 °C, 1333 Pa [cite: 63, 209, 210, 211]
W0 = 295.235
W1 = 2.6422
W2 = -0.032380
W3 = 0.004028
# Correction factor for water vapor refractivity [cite: 64]
CF_VAPOR = 1.022

# --- BIPM-91 Density Equation Constants ---
# Saturation Vapor Pressure (SVP) [cite: 214, 215]
A_SVP = 1.2378847e-5  # K⁻²
B_SVP = -1.9121316e-2  # K⁻¹
C_SVP = 33.93711047
D_SVP = -6.3431645e3  # K

# Enhancement Factor (f) [cite: 217]
ALPHA_F = 1.00062
BETA_F = 3.14e-8  # Pa⁻¹
GAMMA_F = 5.6e-7  # °C⁻²

# Compressibility (Z) [cite: 222, 223, 224, 225, 226, 227]
A0_Z = 1.58123e-6  # K Pa⁻¹
A1_Z = -2.9331e-8  # Pa⁻¹
A2_Z = 1.1043e-10  # K⁻¹ Pa⁻¹
B0_Z = 5.707e-6  # K Pa⁻¹
B1_Z = -2.051e-8  # Pa⁻¹
C0_Z = 1.9898e-4  # K Pa⁻¹
C1_Z = -2.376e-6  # Pa⁻¹
D_Z = 1.83e-11  # K² Pa⁻²
E_Z = -0.765e-8  # K² Pa⁻²


def _calculate_molar_mass_air(co2_ppm: float) -> float:
    """Calculates the molar mass of dry air based on CO₂ concentration."""
    # Source: Ciddor (1996), text following Eq. (4) [cite: 77]
    return 1e-3 * (28.9635 + 12.011e-6 * (co2_ppm - CO2_MOLAR_PPM))


def _calculate_saturation_vapor_pressure(temp_k: float) -> float:
    """Calculates saturation vapor pressure of water."""
    # Source: Ciddor (1996), Appendix A [cite: 213, 214, 215]
    return np.exp(A_SVP * temp_k**2 + B_SVP * temp_k + C_SVP + D_SVP / temp_k)


def _calculate_enhancement_factor(pressure_pa: float, temp_c: float) -> float:
    """Calculates the enhancement factor of water vapor in air."""
    # Source: Ciddor (1996), Appendix A [cite: 78, 216, 217]
    return ALPHA_F + BETA_F * pressure_pa + GAMMA_F * temp_c**2


def _calculate_compressibility(
    pressure_pa: float, temp_k: float, molar_fraction_h2o: float
) -> float:
    """Calculates the compressibility factor Z for moist air."""
    # Source: Ciddor (1996), Appendix A, Eq. (12) [cite: 219, 221]
    t_c = temp_k - 273.15
    xw = molar_fraction_h2o

    term1 = A0_Z + A1_Z * t_c + A2_Z * t_c**2
    term2 = (B0_Z + B1_Z * t_c) * xw
    term3 = (C0_Z + C1_Z * t_c) * xw**2
    term4 = D_Z + E_Z * xw**2

    z = (
        1.0
        - (pressure_pa / temp_k) * (term1 + term2 + term3)
        + (pressure_pa / temp_k) ** 2 * term4
    )
    return z


def _get_density(
    pressure_pa: float,
    temp_k: float,
    molar_mass_air: float,
    molar_fraction_h2o: float,
) -> float:
    """Calculates the density of a moist air sample using the BIPM-91 equation."""
    # Source: Ciddor (1996), Eq. (4) [cite: 74]
    z = _calculate_compressibility(pressure_pa, temp_k, molar_fraction_h2o)
    if z == 0:
        return 0.0

    # Density of the full moist air sample
    rho = (pressure_pa * molar_mass_air / (z * R_GAS_CONSTANT * temp_k)) * (
        1.0 - molar_fraction_h2o * (1.0 - M_W_VAPOR / molar_mass_air)
    )
    return rho


def ciddor_refractive_index(
    wavelength_um: float, conditions: EnvironmentalConditions
) -> float:
    """Calculates the refractive index of air using the full Ciddor (1996) model.

    This implementation follows the detailed procedure outlined in Appendix B
    of the paper[cite: 231]. It calculates component densities and scales standard
    refractivities to find the final refractive index of moist air.

    Args:
        wavelength_um: The wavelength of light in a vacuum, in micrometers (μm).
        conditions: An EnvironmentalConditions object with temperature, pressure,
            relative humidity, and CO₂ concentration.

    Returns:
        The phase refractive index of air (n).
    """
    # --- 1. Calculate vacuum wavenumber and standard refractivities ---
    sigma_sq = (1.0 / wavelength_um) ** 2

    # Refractivity of dry air at 15°C, 101325 Pa, 450 ppm CO₂ [cite: 56]
    n_as_minus_1 = 1e-8 * (K1 / (K0 - sigma_sq) + K3 / (K2 - sigma_sq))

    # Correct for actual CO₂ concentration
    n_axs_minus_1 = n_as_minus_1 * (
        1.0 + CO2_CORR_FACTOR * (conditions.co2_ppm - CO2_STD_PPM)
    )

    # Refractivity of pure water vapor at 20°C, 1333 Pa [cite: 63]
    n_ws_minus_1 = (
        1e-8 * CF_VAPOR * (W0 + W1 * sigma_sq + W2 * sigma_sq**2 + W3 * sigma_sq**3)
    )

    # --- 2. Calculate actual and standard densities via BIPM-91 ---
    m_a = _calculate_molar_mass_air(conditions.co2_ppm)

    # Density of standard dry air (at actual CO₂ level)
    rho_axs = _get_density(P_STD_AIR_PA, T_STD_AIR_K, m_a, 0.0)

    # Density of standard pure water vapor
    # For pure vapor, molar fraction xw=1, and molar mass is M_W_VAPOR
    rho_ws = (P_STD_VAPOR_PA * M_W_VAPOR) / (
        _calculate_compressibility(P_STD_VAPOR_PA, T_STD_VAPOR_K, 1.0)
        * R_GAS_CONSTANT
        * T_STD_VAPOR_K
    )

    # Properties of the actual moist air sample
    t_c = conditions.temperature
    t_k = t_c + 273.15
    p_pa = conditions.pressure
    rh = conditions.relative_humidity

    svp = _calculate_saturation_vapor_pressure(t_k)
    f = _calculate_enhancement_factor(p_pa, t_c)
    xw = f * rh * svp / p_pa if p_pa > 0 else 0.0

    z_actual = _calculate_compressibility(p_pa, t_k, xw)

    # Density of the dry air component in the moist air sample [cite: 242]
    rho_a = (p_pa * m_a * (1.0 - xw)) / (z_actual * R_GAS_CONSTANT * t_k)

    # Density of the water vapor component in the moist air sample [cite: 243, 244]
    rho_w = (p_pa * M_W_VAPOR * xw) / (z_actual * R_GAS_CONSTANT * t_k)

    # --- 3. Combine refractivities scaled by density ratios ---
    # Source: Ciddor (1996), Eq. (5)
    term_air = (rho_a / rho_axs) * n_axs_minus_1 if rho_axs > 0 else 0.0
    term_vapor = (rho_w / rho_ws) * n_ws_minus_1 if rho_ws > 0 else 0.0

    n_final_minus_1 = term_air + term_vapor

    return 1.0 + n_final_minus_1
