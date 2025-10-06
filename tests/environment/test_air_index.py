# tests/environment/test_air_index.py
"""
Unit tests for the main air refractive index dispatcher function.

This file tests that the `refractive_index_air` function correctly dispatches
to the appropriate underlying model (e.g., Ciddor, Edlen) and handles various
input validation and error conditions.
"""
from __future__ import annotations

import pytest

from optiland.environment.air_index import refractive_index_air
from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.birch_downs import birch_downs_refractive_index
from optiland.environment.models.ciddor import ciddor_refractive_index
from optiland.environment.models.edlen import edlen_refractive_index
from optiland.environment.models.kohlrausch import kohlrausch_refractive_index


@pytest.fixture
def typical_conditions() -> EnvironmentalConditions:
    """Provides a standard set of environmental conditions for testing."""
    return EnvironmentalConditions(
        temperature=20.0, pressure=100000.0, relative_humidity=0.5, co2_ppm=400.0
    )


@pytest.fixture
def reference_wavelength_um() -> float:
    """Provides a standard reference wavelength in micrometers."""
    return 0.55


@pytest.mark.parametrize(
    "model_name, model_func",
    [
        ("ciddor", ciddor_refractive_index),
        ("edlen", edlen_refractive_index),
        ("birch_downs", birch_downs_refractive_index),
        ("kohlrausch", kohlrausch_refractive_index),
    ],
)
def test_dispatch_to_correct_model(
    typical_conditions, reference_wavelength_um, model_name, model_func, set_test_backend
):
    """
    Tests that the main `refractive_index_air` function correctly dispatches
    the calculation to the specified underlying model function.
    """
    expected_n = model_func(reference_wavelength_um, typical_conditions)
    n_via_dispatcher = refractive_index_air(
        reference_wavelength_um, typical_conditions, model=model_name
    )
    assert n_via_dispatcher == pytest.approx(expected_n)

    # Test that the model name is case-insensitive
    n_via_dispatcher_upper = refractive_index_air(
        reference_wavelength_um, typical_conditions, model=model_name.upper()
    )
    assert n_via_dispatcher_upper == pytest.approx(expected_n)


def test_unsupported_model_name(
    typical_conditions, reference_wavelength_um, set_test_backend
):
    """
    Tests that a ValueError is raised when an unsupported model name is provided.
    """
    with pytest.raises(ValueError, match="Unsupported air refractive index model: foobar"):
        refractive_index_air(
            reference_wavelength_um, typical_conditions, model="foobar"
        )


def test_invalid_conditions_type(reference_wavelength_um, set_test_backend):
    """
    Tests that a TypeError is raised if the 'conditions' argument is not an
    instance of the EnvironmentalConditions class.
    """
    with pytest.raises(TypeError, match="must be an instance of EnvironmentalConditions"):
        refractive_index_air(reference_wavelength_um, {"temp": 20}, model="ciddor")


def test_wavelength_validation_passed_to_models(typical_conditions, set_test_backend):
    """
    Tests that input validation within the specific models (e.g., for wavelength)
    is correctly triggered when called via the main dispatcher.
    """
    # Birch & Downs model requires a positive wavelength
    with pytest.raises(ValueError, match="Wavelength must be positive"):
        refractive_index_air(0.0, typical_conditions, model="birch_downs")

    # Kohlrausch model requires a non-zero wavelength
    with pytest.raises(ValueError, match="Wavelength must be non-zero"):
        refractive_index_air(0.0, typical_conditions, model="kohlrausch")


def test_temperature_validation_passed_to_models(reference_wavelength_um, set_test_backend):
    """
    Tests that temperature validation from underlying models is correctly
    triggered, for example, preventing a division-by-zero error in the
    Kohlrausch model at a critical temperature.
    """
    # This temperature causes the denominator in the Kohlrausch formula to be zero
    critical_temp = 15.0 - (1.0 / 3.4785e-3)
    conditions_bad_temp = EnvironmentalConditions(temperature=critical_temp)

    with pytest.raises(ValueError, match="non-positive denominator"):
        refractive_index_air(
            reference_wavelength_um, conditions_bad_temp, model="kohlrausch"
        )