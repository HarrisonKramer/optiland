"""Kohlrausch (1968) Refractive Index Model for Air

This module provides a function to calculate the refractive index of air
according to the specific formula used by Zemax OpticStudio. This formula is
attributed to F. Kohlrausch, Praktische Physik, 1968.

The model consists of a Sellmeier-type dispersion equation for a reference
condition (n_ref) and a linearized scaling factor to account for variations
in temperature and pressure. It is a model for dry air and does not
explicitly account for humidity or CO₂ variations.

References:
    - Kohlrausch, F. (1968). Praktische Physik (Vol. 1, p. 408).

Kramer Harrison, 2025
"""

import optiland.backend as be
from ..conditions import EnvironmentalConditions

# --- Model Constants from the Kohlrausch Formula ---

# Constants for the reference dispersion formula (n_ref).
# This is a Sellmeier-2 formula written as:
# (n_ref - 1) * 10^8 = A + B / (C - σ²) + D / (E - σ²)
# where σ = 1/λ is the wavenumber in μm⁻¹.
DISP_A = 6432.8
DISP_B = 2949810.0
DISP_C = 146.0  # In μm⁻²
DISP_D = 25540.0
DISP_E = 41.0  # In μm⁻²

# Reference and scaling constants for the pressure/temperature correction.
T_REF_C = 15.0  # Reference temperature in degrees Celsius.
P_STD_PA = 101325.0  # Standard atmospheric pressure in Pascals.

# Linearized thermal coefficient for air.
# This approximates ideal gas law scaling around 15°C.
ALPHA_T = 3.4785e-3  # In °C⁻¹


def kohlrausch_air_refractive_index(
    wavelength_um: float, conditions: EnvironmentalConditions
) -> float:
    """Calculates air refractive index using the Kohlrausch formula.

    This model ignores the humidity and CO₂ concentration from the conditions
    object.

    Args:
        wavelength_um: Wavelength of light in a vacuum, in micrometers (μm).
        conditions: An EnvironmentalConditions object. Only temperature and
            pressure are used.

    Returns:
        The refractive index of air (n).

    Raises:
        ValueError: If the denominator of the scaling term becomes non-positive,
            which can occur at extremely low temperatures.
    """
    # --- 1. Calculate the reference refractivity (n_ref - 1) ---
    # The formula uses λ directly, but we convert it to the standard Sellmeier
    # form which uses wavenumber σ = 1/λ for clarity and robustness.
    try:
        sigma_sq = (1.0 / wavelength_um) ** 2
    except ZeroDivisionError as err:
        raise ValueError("Wavelength must be non-zero.") from err

    # Calculate (n_ref - 1) * 10^8
    n_ref_minus_1_e8 = (
        DISP_A + DISP_B / (DISP_C - sigma_sq) + DISP_D / (DISP_E - sigma_sq)
    )
    n_ref_minus_1 = n_ref_minus_1_e8 * 1.0e-8

    # --- 2. Apply temperature and pressure scaling ---
    t_c = conditions.temperature
    p_pa = conditions.pressure

    # P is relative pressure (dimensionless).
    relative_pressure = p_pa / P_STD_PA

    # Denominator of the scaling term.
    temp_scaling_denom = 1.0 + (t_c - T_REF_C) * ALPHA_T
    if temp_scaling_denom <= 0:
        raise ValueError(
            f"Invalid temperature {t_c}°C results in non-positive denominator."
        )

    # Final calculation for n_air
    n_air = 1.0 + (n_ref_minus_1 * relative_pressure) / temp_scaling_denom

    return n_air
