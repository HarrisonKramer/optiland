"""Implementation of the Ciddor (1996) model for the refractive index of air.

This module provides functions to calculate the refractive index of air
based on the Ciddor (1996) formulation, which is an update to the Edlén
equations. It accounts for variations in pressure, temperature, humidity,
and CO2 concentration.

References:
    Ciddor, P. E. (1996). Refractive index of air: new equations for the
    visible and near infrared. Applied Optics, 35(9), 1566-1573.
"""
import math

from optiland.environment.conditions import EnvironmentalConditions

# Constants from Ciddor (1996) for dry air refractivity at 15C, 101325Pa, 400ppm CO2
K0 = 238.0185  # um^-2
K1 = 5792105.0  # unitless, for (n-1)*10^8
K2 = 57.362  # um^-2
K3 = 167917.0  # unitless, for (n-1)*10^8

# Constants for saturation vapor pressure (Appendix A, Ciddor 1996)
# P_sv = exp(A*T_k^2 + B*T_k + C + D/T_k) for T_k in Kelvin, P_sv in Pa
A_svp = 1.2378847e-5  # K^-2
B_svp = -1.9121316e-2  # K^-1
C_svp = 33.93711047   # unitless
D_svp = -6.3431645e3  # K

# Constants for water vapor refractivity term (Eq. 7, Ciddor 1996)
# (n_w-1)*10^8 = p_v * G(t) * [AW + BW*sig^2 + CW*sig^4 + DW*sig^6]
AW_c = 3.0173017    # Pa^-1 K (scaled, see usage)
BW_c = 0.026993284  # Pa^-1 K um^2 (scaled)
CW_c = -0.0003309236 # Pa^-1 K um^4 (scaled)
DW_c = 0.000004116616# Pa^-1 K um^6 (scaled)
ALPHA_E_c = 1.00e-5  # K^-1 (enhancement factor coefficient for water vapor)
T0_WV_c = 20.0       # Celsius (reference temperature for water vapor term)

# Constants for compressibility factor Z_s/Z_a (Eq. 4, Ciddor 1996)
# For [1 + P_s(a0 + a1*t_s + a2*t_s^2)] / [1 + P(a0 + a1*t + a2*t^2)]
A0_comp = 1.58123e-6  # Pa^-1
A1_comp = -2.9331e-8  # Pa^-1 C^-1
A2_comp = 1.1043e-10  # Pa^-1 C^-2

# CO2 correction factor constant (Table 3, Ciddor 1996)
# (n(P,T,x_c) - 1) = (n(P,T,400) - 1) * [1 + CO2_K * (x_c - 400)]
CO2_K = 0.5327e-6  # ppm^-1


def saturation_vapor_pressure(temperature_c):
    """Calculates saturation vapor pressure of water.

    Uses the equation from Appendix A of Ciddor (1996).

    Args:
        temperature_c (float): Temperature in degrees Celsius.

    Returns:
        float: Saturation vapor pressure in Pascals (Pa).
    """
    t_k = temperature_c + 273.15  # Convert to Kelvin
    psv = math.exp(A_svp * t_k**2 + B_svp * t_k + C_svp + D_svp / t_k)
    return psv


def partial_pressure_water_vapor(temperature_c, relative_humidity, pressure_pa):
    """Calculates partial pressure of water vapor.

    Includes the enhancement factor f_w(t,p) as per Ciddor (1996) Eq. 10.

    Args:
        temperature_c (float): Temperature in degrees Celsius.
        relative_humidity (float): Relative humidity (0 to 1).
        pressure_pa (float): Total air pressure in Pascals.

    Returns:
        float: Partial pressure of water vapor in Pascals (Pa).
    """
    psv = saturation_vapor_pressure(temperature_c)
    # Enhancement factor f_w(t,p) from Ciddor (1996) Eq. 10
    f_w = 1.00062 + (3.14e-8 * pressure_pa) + (5.6e-7 * temperature_c**2)
    pv = relative_humidity * f_w * psv
    return pv


def refractivity_dry_air(wavelength_um, pressure_pa, temperature_c, co2_ppm):
    """Calculates the refractivity of dry air component (N_a).

    Implements Equations 2, 3, 4, and 5 from Ciddor (1996).
    Refractivity N = (n-1) * 10^6.

    Args:
        wavelength_um (float): Wavelength of light in microns.
        pressure_pa (float): Total air pressure in Pascals.
        temperature_c (float): Temperature in degrees Celsius.
        co2_ppm (float): CO2 concentration in parts per million.

    Returns:
        float: Refractivity of the dry air component.
    """
    sigma = 1.0 / wavelength_um  # Wavenumber in um^-1
    sigma2 = sigma**2

    # (n_s - 1) for standard air (15C, 101325Pa, 0%RH, 400ppm CO2)
    # From Ciddor Eq. 2: (n_s - 1) * 10^8 = ...
    n_s_minus_1_times_10_8 = (K1 / (K0 - sigma2)) + (K3 / (K2 - sigma2))
    n_s_minus_1 = n_s_minus_1_times_10_8 * 1.0e-8

    # Apply CO2 correction (Eq. 5, Ciddor 1996)
    # This adjusts (n_s-1) from 400ppm reference to actual co2_ppm
    n_s_minus_1_co2_corrected = n_s_minus_1 * (1 + CO2_K * (co2_ppm - 400.0))

    # Apply pressure and temperature correction (Eq. 3 and 4, Ciddor 1996)
    t_k = temperature_c + 273.15  # Current temperature in Kelvin
    t_s_k = 15.0 + 273.15         # Standard temperature (15 C) in Kelvin
    p_s_pa = 101325.0             # Standard pressure in Pascals

    # Compressibility factor ratio Z_s/Z_a from Eq. 4
    term_s_comp = 1 + p_s_pa * (A0_comp + A1_comp * 15.0 + A2_comp * 15.0**2)
    term_actual_comp = 1 + pressure_pa * \
        (A0_comp + A1_comp * temperature_c + A2_comp * temperature_c**2)
    compressibility_ratio = term_s_comp / term_actual_comp

    # (n_a - 1) for actual P, T, xc (Eq. 3)
    n_dry_air_minus_1 = n_s_minus_1_co2_corrected * \
                        (pressure_pa / p_s_pa) * (t_s_k / t_k) * \
                        compressibility_ratio

    return n_dry_air_minus_1 * 1.0e6 # Return as refractivity (n-1)*10^6


def refractivity_water_vapor_term(wavelength_um, temperature_c, partial_pressure_pv_pa):
    """Calculates the water vapor refractivity term (N_wp).

    Implements Equation 7 from Ciddor (1996). This term is subtracted.
    Refractivity N = (n-1) * 10^6.

    Args:
        wavelength_um (float): Wavelength of light in microns.
        temperature_c (float): Temperature in degrees Celsius.
        partial_pressure_pv_pa (float): Partial pressure of water vapor in Pa.

    Returns:
        float: Refractivity term due to water vapor.
    """
    if partial_pressure_pv_pa == 0: # No humidity, no water vapor term
        return 0.0

    sigma = 1.0 / wavelength_um
    sigma2 = sigma**2
    t_k = temperature_c + 273.15

    # G(t) factor from Eq. 7
    g_t = (1.0 / t_k) * (1 + ALPHA_E_c * (temperature_c - T0_WV_c))

    # Spectral dispersion term for water vapor from Eq. 7
    spectral_term_wv = AW_c + BW_c * sigma2 + CW_c * sigma2**2 + DW_c * sigma2**3

    # (n_wp - 1) * 10^8 (as per Ciddor's constant scaling for A_w, B_w etc.)
    n_wp_minus_1_times_10_8 = partial_pressure_pv_pa * g_t * spectral_term_wv

    # Convert to (n-1)*10^6 for consistency with other refractivity terms
    n_wp_refractivity = n_wp_minus_1_times_10_8 * 1.0e-2
    return n_wp_refractivity


def ciddor_refractive_index(wavelength_um, conditions):
    """Calculates the refractive index of air using the Ciddor (1996) model.

    Args:
        wavelength_um (float): Wavelength of light in microns (μm).
        conditions (EnvironmentalConditions): An object holding the
            environmental parameters (pressure, temperature, humidity, CO2 ppm).

    Returns:
        float: The refractive index of air (n).

    Raises:
        TypeError: If conditions is not an EnvironmentalConditions object.
    """
    if not isinstance(conditions, EnvironmentalConditions):
        raise TypeError("conditions must be an EnvironmentalConditions object.")

    # Ciddor (1996) states validity for 0.3 um to 1.69 um.
    # No explicit error/warning here, but users should be aware.

    pv_pa = partial_pressure_water_vapor(
        conditions.temperature,
        conditions.relative_humidity,
        conditions.pressure
    )

    # Refractivity of the dry air component at total pressure P
    # (n_a(P,T,x_c) - 1) * 10^6
    n_a_refractivity = refractivity_dry_air(
        wavelength_um,
        conditions.pressure, # Total pressure
        conditions.temperature,
        conditions.co2_ppm
    )

    # Refractivity term due to water vapor
    # (n_wp(sigma, T, p_v) - 1) * 10^6
    n_wp_refractivity_term = refractivity_water_vapor_term(
        wavelength_um,
        conditions.temperature,
        pv_pa
    )

    # Total refractivity of moist air (Eq. 1, Ciddor 1996)
    # (n_moist_air - 1) * 10^6 = N_a - N_wp
    total_refractivity = n_a_refractivity - n_wp_refractivity_term

    n = 1.0 + total_refractivity * 1.0e-6
    return n
