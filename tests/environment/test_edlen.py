# tests/environment/test_edlen.py
"""
Tests for the Edlen (1966) dispersion formula for the refractive index of air.
"""
from __future__ import annotations
from math import isclose

import pytest

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.edlen import (
    _calculate_saturation_vapor_pressure,
    edlen_refractive_index,
)


class TestEdlenRefractiveIndex:
    """
    Tests the Edlen (1966) model for calculating the refractive index of air,
    including its helper functions and behavior under various environmental
    conditions.
    """

    STD_TEMP_C = 15.0
    STD_PRES_PA = 101325.0
    STD_RH = 0.0
    STD_CO2_PPM_EDLEN = 300.0
    REF_WAVELENGTH_UM = 0.632991
    REF_N_MINUS_1_EDLEN = 2.764783e-4

    def setup_method(self):
        """
        Sets up a standard set of environmental conditions based on the
        Edlen model's reference.
        """
        self.std_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM_EDLEN,
        )

    def test_calculate_saturation_vapor_pressure(self):
        """
        Tests the helper function for calculating saturation vapor pressure
        against a known value from the Buck (1981) formula.
        """
        # Value for 15C from Buck (1981) formula
        expected_svp_15c = 1705.325
        assert isclose(
            _calculate_saturation_vapor_pressure(15.0), expected_svp_15c, rel_tol=1e-5
        )

    def test_edlen_refractive_index_nist_reference(self):
        """
        Tests the Edlen model against a reference value from NIST documentation.
        """
        nist_conditions = EnvironmentalConditions(
            temperature=20.0, pressure=101325.0, relative_humidity=0.0, co2_ppm=450.0
        )
        expected_n = 1.000271759
        n_edlen = edlen_refractive_index(0.633, nist_conditions)
        assert isclose(n_edlen, expected_n, rel_tol=1e-9)

    def test_edlen_refractive_index_effect_of_co2(self):
        """
        Verifies that increasing CO2 concentration increases the refractive index.
        """
        conditions_high_co2 = EnvironmentalConditions(co2_ppm=1000.0)
        n_high_co2 = edlen_refractive_index(self.REF_WAVELENGTH_UM, conditions_high_co2)
        assert n_high_co2 > (1.0 + self.REF_N_MINUS_1_EDLEN)

    def test_edlen_refractive_index_effect_of_humidity(self):
        """
        Verifies that increasing humidity decreases the refractive index.
        """
        moist_conditions = EnvironmentalConditions(relative_humidity=0.8)
        dry_conditions = EnvironmentalConditions(relative_humidity=0.0)
        n_moist = edlen_refractive_index(0.55, moist_conditions)
        n_dry = edlen_refractive_index(0.55, dry_conditions)
        assert n_moist < n_dry

    def test_edlen_refractive_index_effect_of_temperature(self):
        """
        Verifies that increasing temperature decreases the refractive index.
        """
        cold_conditions = EnvironmentalConditions(temperature=0.0)
        hot_conditions = EnvironmentalConditions(temperature=30.0)
        n_cold = edlen_refractive_index(self.REF_WAVELENGTH_UM, cold_conditions)
        n_hot = edlen_refractive_index(self.REF_WAVELENGTH_UM, hot_conditions)
        assert n_cold > n_hot

    def test_edlen_refractive_index_effect_of_pressure(self):
        """
        Verifies that increasing pressure increases the refractive index.
        """
        low_p_conditions = EnvironmentalConditions(pressure=50000.0)
        high_p_conditions = EnvironmentalConditions(pressure=200000.0)
        n_low_p = edlen_refractive_index(self.REF_WAVELENGTH_UM, low_p_conditions)
        n_high_p = edlen_refractive_index(self.REF_WAVELENGTH_UM, high_p_conditions)
        assert n_low_p < n_high_p

    def test_edlen_refractive_index_effect_of_wavelength(self):
        """
        Verifies the dispersion of the model (shorter wavelength gives
        higher refractive index).
        """
        n_blue = edlen_refractive_index(0.4, self.std_conditions)
        n_red = edlen_refractive_index(0.7, self.std_conditions)
        assert n_blue > n_red

    def test_edlen_refractive_index_zero_pressure(self):
        """
        Tests that at zero pressure, the refractive index is 1 (vacuum).
        """
        conditions_vacuum = EnvironmentalConditions(pressure=0.0)
        n_vacuum = edlen_refractive_index(self.REF_WAVELENGTH_UM, conditions_vacuum)
        assert isclose(n_vacuum, 1.0, abs_tol=1e-9)

    def test_edlen_refractive_index_invalid_wavelength(self):
        """
        Tests that a non-positive wavelength raises a ValueError.
        """
        with pytest.raises(ValueError, match="Wavelength must be positive."):
            edlen_refractive_index(0.0, self.std_conditions)

    def test_edlen_refractive_index_invalid_conditions_type(self):
        """
        Tests that providing an invalid type for the `conditions` argument
        raises a TypeError.
        """
        with pytest.raises(TypeError, match="conditions must be an EnvironmentalConditions object."):
            edlen_refractive_index(self.REF_WAVELENGTH_UM, "not_a_condition_object")