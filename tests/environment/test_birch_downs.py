# tests/environment/test_birch_downs.py
"""
Tests for the Birch and Downs (1993, 1994) dispersion formula for the
refractive index of air.
"""
import pytest

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.birch_downs import birch_downs_refractive_index
from ..utils import assert_allclose


@pytest.fixture
def typical_conditions() -> EnvironmentalConditions:
    """Provides a standard set of environmental conditions for testing."""
    return EnvironmentalConditions(
        temperature=20.0, pressure=100000.0, relative_humidity=0.5, co2_ppm=400.0
    )


class TestBirchDownsRefractiveIndex:
    """
    Tests the Birch and Downs model for calculating the refractive index of air.
    """

    def test_non_positive_wavelength_raises_error(self, typical_conditions, set_test_backend):
        """
        Tests that a non-positive wavelength input raises a ValueError.
        """
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            birch_downs_refractive_index(0.0, typical_conditions)
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            birch_downs_refractive_index(-0.55, typical_conditions)

    def test_temperature_out_of_range_warns(self, reference_wavelength_um, set_test_backend):
        """
        Tests that a warning is issued if the temperature is outside the
        model's recommended range.
        """
        bad_conditions = EnvironmentalConditions(temperature=-50.0)  # Below range
        with pytest.warns(UserWarning, match="Temperature is outside"):
            birch_downs_refractive_index(reference_wavelength_um, bad_conditions)

    def test_pressure_out_of_range_warns(self, reference_wavelength_um, set_test_backend):
        """
        Tests that a warning is issued if the pressure is outside the
        model's recommended range.
        """
        bad_conditions = EnvironmentalConditions(pressure=50000.0)  # Below range
        with pytest.warns(UserWarning, match="Pressure is outside"):
            birch_downs_refractive_index(reference_wavelength_um, bad_conditions)

    def test_humidity_out_of_range_warns(self, reference_wavelength_um, set_test_backend):
        """
        Tests that a warning is issued if the humidity is outside the
        model's recommended range.
        """
        bad_conditions = EnvironmentalConditions(relative_humidity=1.5)  # Above range
        with pytest.warns(UserWarning, match="Relative humidity is outside"):
            birch_downs_refractive_index(reference_wavelength_um, bad_conditions)

    def test_co2_ppm_out_of_range_warns(self, reference_wavelength_um, set_test_backend):
        """
        Tests that a warning is issued if the CO2 concentration is outside the
        model's recommended range.
        """
        bad_conditions = EnvironmentalConditions(co2_ppm=1500)  # Above range
        with pytest.warns(UserWarning, match="CO2 concentration is outside"):
            birch_downs_refractive_index(reference_wavelength_um, bad_conditions)

    @pytest.mark.parametrize(
        "wavelength, temp, pressure, humidity, co2_ppm, expected_n",
        [
            # Reference value from external calculator for standard conditions
            (0.589, 15.0, 101325.0, 0.0, 450.0, 1.00027815),
            # Test at different temperature
            (0.633, 25.0, 101325.0, 0.5, 400.0, 1.00026629),
            # Test at different pressure
            (0.550, 20.0, 95000.0, 0.5, 400.0, 1.00025644),
        ],
    )
    def test_calculation_against_reference_values(
        self, wavelength, temp, pressure, humidity, co2_ppm, expected_n, set_test_backend
    ):
        """
        Tests the refractive index calculation against known reference values
        for a variety of environmental conditions.
        """
        conditions = EnvironmentalConditions(
            temperature=temp,
            pressure=pressure,
            relative_humidity=humidity,
            co2_ppm=co2_ppm,
        )
        n = birch_downs_refractive_index(wavelength, conditions)
        assert_allclose(n, expected_n, atol=1e-8)