"""Provides a unified interface for calculating the refractive index of air.

This module offers a single function, `refractive_index_air`, to access
different models for air refractive index based on environmental conditions.
It supports Ciddor (1996), Edlén (1966), Birch & Downs (1993/1994),
and a simplified Kohlrausch model.

Kramer Harrison, 2025
"""

from .birch_downs import birch_downs_refractive_index
from .ciddor import ciddor_refractive_index
from .conditions import EnvironmentalConditions
from .edlen import edlen_refractive_index
from .kohlrausch import kohlrausch_refractive_index


def refractive_index_air(wavelength_um, conditions, model="ciddor"):
    """Calculates the refractive index of air using a specified model.

    This function acts as a dispatcher to various air refractive index models.
    The environmental conditions (temperature, pressure, humidity, CO2) are
    encapsulated in the `conditions` object.

    Args:
        wavelength_um (float): The wavelength of light in microns (μm).
        conditions (EnvironmentalConditions): An object containing the
            environmental parameters. Specific models might ignore some
            parameters (e.g., simplified Kohlrausch ignores humidity and CO2).
            - pressure (Pa)
            - temperature (°C)
            - relative_humidity (0 to 1)
            - co2_ppm (parts per million)
        model (str, optional): The model to use for calculation.
            Supported models: 'ciddor', 'edlen', 'birch_downs', 'kohlrausch'.
            Defaults to 'ciddor'.

    Returns:
        float: The calculated refractive index of air.

    Raises:
        ValueError: If an unsupported model is specified.
        TypeError: If `conditions` is not an `EnvironmentalConditions` object.

    Example:
        >>> from optiland.environment import EnvironmentalConditions
        >>> conditions_std = EnvironmentalConditions(temperature=15.0,
        ...                                          pressure=101325.0,
        ...                                          relative_humidity=0.0,
        ...                                          co2_ppm=400.0)
        >>> n_ciddor = refractive_index_air(0.55, conditions_std, model='ciddor')
        >>> print(f"Ciddor n(0.55um, std cond): {n_ciddor:.9f}")
        Ciddor n(0.55um, std cond): 1.000277641
    """
    if not isinstance(conditions, EnvironmentalConditions):
        raise TypeError(
            "Input 'conditions' must be an instance of EnvironmentalConditions."
        )

    model_lower = model.lower()
    if model_lower == "ciddor":
        return ciddor_refractive_index(wavelength_um, conditions)
    elif model_lower == "edlen":
        return edlen_refractive_index(wavelength_um, conditions)
    elif model_lower == "birch_downs":
        return birch_downs_refractive_index(wavelength_um, conditions)
    elif model_lower == "kohlrausch":
        return kohlrausch_refractive_index(wavelength_um, conditions)
    else:
        raise ValueError(
            f"Unsupported air refractive index model: {model}. "
            "Supported models are: 'ciddor', 'edlen', 'birch_downs', 'kohlrausch'."
        )
