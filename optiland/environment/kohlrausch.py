"""Implementation of a simplified Kohlrausch model for the refractive index of air.

This module provides a basic calculation for the refractive index of air based
on principles described in older physics handbooks like Kohlrausch's
Praktische Physik. It typically involves a dispersion formula for reference
conditions and scaling factors for temperature and pressure.

This implementation is for dry air and does not account for humidity or CO2
variations with the same detail as modern models like Ciddor or Edlén.
It's provided for historical context or when a simpler approximation is needed.

A common form for dry air at 0°C and 101325 Pa is:
  (n_0 - 1) * 10^6 = A + B / lambda^2
And then scaled for actual temperature and pressure:
  (n - 1) = (n_0 - 1) * (P / P_0) * (T_0 / T)
"""

import math
from optiland.environment.conditions import EnvironmentalConditions

# Constants for a simplified Cauchy-like dispersion formula for dry air
# at T0 = 0°C (273.15 K) and P0 = 101325 Pa.
# (n0 - 1) * 10^6 = A_k + B_k / lambda^2
# These values are approximate and can vary in different summaries of
# older models. These are chosen to be representative.
A_K = 287.5  # Dimensionless, for (n0-1)*10^6
B_K = 5.0    # um^2, for (n0-1)*10^6 (if lambda is in um)

T0_KELVIN = 273.15  # 0 degrees Celsius in Kelvin
P0_PASCAL = 101325.0 # Standard pressure in Pascals


def kohlrausch_refractive_index(wavelength_um, conditions):
    """Calculates the refractive index of dry air using a simplified Kohlrausch model.

    The model uses a basic dispersion formula for air at 0°C and 101325 Pa,
    and then scales this value for the given temperature and pressure.
    Humidity and CO2 concentration from the conditions object are ignored
    by this simplified model.

    Args:
        wavelength_um (float): Wavelength of light in microns (μm).
        conditions (EnvironmentalConditions): An object holding the
            environmental parameters. Only pressure and temperature are used.

    Returns:
        float: The refractive index of dry air (n).

    Raises:
        TypeError: If conditions is not an EnvironmentalConditions object.
        ValueError: If wavelength_um is not positive.
    """
    if not isinstance(conditions, EnvironmentalConditions):
        raise TypeError("conditions must be an EnvironmentalConditions object.")
    if not wavelength_um > 0:
        raise ValueError("Wavelength must be positive.")

    # Calculate refractivity at reference conditions (0°C, 101325 Pa)
    # (n0 - 1) * 10^6 = A_K + B_K / (wavelength_um^2)
    refractivity0_times_10_6 = A_K + B_K / (wavelength_um**2)
    n0_minus_1 = refractivity0_times_10_6 * 1.0e-6

    # Get current temperature in Kelvin and pressure in Pascals
    t_k_actual = conditions.temperature + 273.15
    p_actual_pa = conditions.pressure

    if t_k_actual <= 0:
        raise ValueError("Absolute temperature must be positive.")

    # Scale refractivity for actual temperature and pressure
    # (n - 1) = (n0 - 1) * (P_actual / P0) * (T0 / T_actual)
    n_minus_1_actual = n0_minus_1 * \
                       (p_actual_pa / P0_PASCAL) * \
                       (T0_KELVIN / t_k_actual)

    n = 1.0 + n_minus_1_actual
    return n
