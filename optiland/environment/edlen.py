"""Implementation of the Edlén (1966) model for the refractive index of air.

This module provides functions to calculate the refractive index of air
based on the Edlén (1966) formulation. It includes dispersion for standard
air and corrections for temperature, pressure, and water vapor.

The original Edlén formulas often used pressure in mmHg (Torr). This
implementation expects SI units (Pascals, Celsius) and handles conversions
internally where necessary.

References:
    Edlén, B. (1966). The Refractive Index of Air. Metrologia, 2(2), 71-80.
"""
import math
from optiland.environment.conditions import EnvironmentalConditions

# Constants for Edlén (1966) dispersion formula for standard air
# Standard air: 15 °C, 101325 Pa (760 mmHg), dry, 0.03% CO2 (300 ppm)
# (n_s - 1) * 10^8 = C1 + C2 / (C3 - sigma^2) + C4 / (C5 - sigma^2)
EDLEN_C1 = 8342.13
EDLEN_C2 = 2406030.0
EDLEN_C3 = 130.0
EDLEN_C4 = 15997.0
EDLEN_C5 = 38.9

# Conversion factor: 1 Torr (mmHg) to Pascals
TORR_TO_PA = 101325.0 / 760.0

# Reference temperature for gas law scaling in Edlen's formulation (0 C = 273.15 K)
# The factor (1 + 0.003661*t) implies alpha = 0.003661 /K and T0 = 0 C.
ALPHA_GAS = 0.003661 # /K, thermal expansion coefficient of air relative to 0 C


def _saturation_vapor_pressure_edlen(temperature_c):
    """Calculates saturation vapor pressure using a Magnus-Tetens like formula.
    This is a common approximation used with Edlén-era models.
    Args:
        temperature_c (float): Temperature in degrees Celsius.
    Returns:
        float: Saturation vapor pressure in Pascals (Pa).
    """
    # P_sv_hPa = 6.1078 * 10**( (7.5 * T_c) / (T_c + 237.3) ) (original Magnus-Tetens)
    # More common form (Buck 1981/1996 or similar approximations):
    # P_sv_Pa = 611.2 * exp((17.67 * T_c) / (T_c + 243.5)) (for over water)
    # Edlen's paper refers to tables, so a reasonable approximation is fine.
    # For simplicity, using the one from Ciddor's implementation for now,
    # as it's already vetted, though Edlen might have used a different one.
    # Let's use a standard Magnus formula:
    # svp in hPa = 6.112 * exp((17.67 * T_c)/(T_c + 243.5))
    svp_hpa = 6.112 * math.exp((17.67 * temperature_c) / (temperature_c + 243.5))
    return svp_hpa * 100.0 # Convert hPa to Pa


def edlen_refractive_index(wavelength_um, conditions):
    """Calculates air refractive index using Edlén (1966) equations.

    Args:
        wavelength_um (float): Wavelength of light in microns (μm).
        conditions (EnvironmentalConditions): Environmental parameters.
            CO2 concentration is implicitly assumed to be 300 ppm as per
            Edlén's standard air definition.

    Returns:
        float: Refractive index of air (n).
    """
    if not isinstance(conditions, EnvironmentalConditions):
        raise TypeError("conditions must be an EnvironmentalConditions object.")
    if not wavelength_um > 0:
        raise ValueError("Wavelength must be positive.")

    sigma = 1.0 / wavelength_um  # Wavenumber in um^-1
    sigma2 = sigma**2

    # 1. Refractivity of standard air (n_s - 1)
    # (15 °C, 101325 Pa (760 mmHg), dry, 300 ppm CO2)
    n_s_minus_1_times_10_8 = EDLEN_C1 + \
                             EDLEN_C2 / (EDLEN_C3 - sigma2) + \
                             EDLEN_C4 / (EDLEN_C5 - sigma2)
    n_s_minus_1 = n_s_minus_1_times_10_8 * 1.0e-8

    # 2. Correction for actual temperature and pressure (for dry air)
    # (n_tp - 1) = (n_s - 1) * K_pt
    # K_pt = [P_torr * (1 + P_torr * (60.1 - 0.972*T_c)*10^-8)] /
    #        [720.775 * (1 + 0.003661*T_c)]
    # This can be rewritten using P_Pa and T_c directly.
    # (n_tp-1) = (n_s-1) * (P_Pa/101325 Pa) * (288.15 K / (T_c+273.15 K)) * CF
    # Where CF is a compressibility correction.
    # Using Edlen's explicit formula structure for K_pt:
    p_torr = conditions.pressure / TORR_TO_PA
    t_c = conditions.temperature

    # Denominator of K_pt: 720.775 * (1 + alpha * t_c)
    # Note: 720.775 is P_s_torr / (1 + alpha * t_s_C) = 760 / (1+0.003661*15)
    # So K_pt simplifies to (P_torr/P_s_torr) * ((1+alpha*t_s_C)/(1+alpha*t_c)) * non-ideal_gas_terms
    # Let's use the form from Ciddor (1996) Appendix C, Eq C2 (Edlen Dry Air):
    # N_d(t,p) = N_s * (p/760) * (1 / (1+ALPHA_GAS*t)) * (1 - p*beta_t)
    # where N_s = (n_s-1)*10^6, p is in Torr
    # beta_t = (0.0624 - 0.000680*t)*1e-6
    # This is (n_tp - 1) * 10^6
    beta_t = (0.0624 - 0.000680 * t_c) * 1.0e-6
    n_tp_minus_1 = (n_s_minus_1 * (p_torr / 760.0) *
                    (1.0 / (1.0 + ALPHA_GAS * t_c)) *
                    (1.0 - p_torr * beta_t))

    # 3. Correction for water vapor partial pressure f (in Torr)
    # This term is subtracted from n_tp, or its refractivity subtracted from
    # the dry air refractivity.
    # Δn_w = - f * (5.722 - 0.0457 * σ²) * 10⁻⁸
    # n_final - 1 = (n_tp - 1) + Δn_w

    svp_pa = _saturation_vapor_pressure_edlen(t_c)
    f_pa = conditions.relative_humidity * svp_pa # Partial pressure in Pa
    f_torr = f_pa / TORR_TO_PA

    delta_n_w = -f_torr * (5.722 - 0.0457 * sigma2) * 1.0e-8

    n_final_minus_1 = n_tp_minus_1 + delta_n_w
    n = 1.0 + n_final_minus_1

    return n
