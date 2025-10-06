# tests/environment/test_ciddor.py
"""
Tests for the Ciddor (1996) dispersion formula for the refractive index of air.
"""
from __future__ import annotations
from unittest.mock import patch

import pytest
from math import exp, isclose

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.ciddor import (
    _calculate_molar_mass_air,
    _calculate_saturation_vapor_pressure,
    _calculate_enhancement_factor,
    _calculate_compressibility,
    ciddor_refractive_index,
    A0_Z, A1_Z, A2_Z, B0_Z, B1_Z, C0_Z, C1_Z, D_Z, E_Z,
    ALPHA_F, BETA_F, GAMMA_F, CO2_MOLAR_PPM
)
from ..utils import assert_allclose


class TestCiddorRefractiveIndex:
    """
    Tests the Ciddor (1996) model for calculating the refractive index of air.
    It includes tests for helper functions and the main calculation under
    various environmental conditions.
    """

    STD_TEMP_C = 15.0
    STD_PRES_PA = 101325.0
    STD_RH = 0.0
    STD_CO2_PPM = 450.0
    REF_WAVELENGTH_UM = 0.6328
    # Reference value from refractiveindex.info for Ciddor model at standard conditions
    REF_N_MINUS_1_CIDDOR = 2.76534e-4

    @pytest.fixture(autouse=True)
    def setup_method(self, set_test_backend):
        """Set up standard conditions for each test."""
        self.std_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
            relative_humidity=self.STD_RH,
            co2_ppm=self.STD_CO2_PPM,
        )

    def test_calculate_molar_mass_air(self, set_test_backend):
        """
        Tests the calculation of the molar mass of air at different CO2
        concentrations.
        """
        expected_mass_400ppm = 1e-3 * (28.9635 + 12.011e-6 * (400.0 - CO2_MOLAR_PPM))
        assert_allclose(_calculate_molar_mass_air(400.0), expected_mass_400ppm)

    def test_calculate_saturation_vapor_pressure(self, set_test_backend):
        """
        Tests the calculation of saturation vapor pressure (SVP) against a
        known value from the Ciddor paper.
        """
        expected_svp_20c = 2338.8  # Pa, from Ciddor (1996), Table A1 for 20C
        assert_allclose(
            _calculate_saturation_vapor_pressure(20.0), expected_svp_20c, rtol=1e-3
        )

    def test_calculate_enhancement_factor(self, set_test_backend):
        """
        Tests the calculation of the water vapor enhancement factor against a
        known value.
        """
        expected_f_20c_1atm = 1.00062 + (3.14e-8 * 101325.0) + (5.6e-7 * 20.0**2)
        assert_allclose(
            _calculate_enhancement_factor(101325.0, 20.0), expected_f_20c_1atm
        )

    def test_calculate_compressibility(self, set_test_backend):
        """
        Tests the calculation of the compressibility factor (Z) for dry and
        moist air.
        """
        # Test for dry air at standard conditions
        expected_z_dry = 1.0 - (101325.0 / 293.15) * (A0_Z + A1_Z * 20.0 + A2_Z * 20.0**2) + (101325.0 / 293.15)**2 * D_Z
        assert_allclose(
            _calculate_compressibility(101325.0, 293.15, 0.0), expected_z_dry
        )

    def test_ciddor_refractive_index_standard_conditions(self, set_test_backend):
        """
        Tests the main Ciddor calculation against a reference value for
        standard conditions.
        """
        n_ciddor = ciddor_refractive_index(self.REF_WAVELENGTH_UM, self.std_conditions)
        assert_allclose(n_ciddor - 1.0, self.REF_N_MINUS_1_CIDDOR, atol=1e-8)

    def test_ciddor_refractive_index_effect_of_co2(self, set_test_backend):
        """
        Verifies that increasing CO2 concentration increases the refractive
        index, and vice versa.
        """
        conditions_high_co2 = EnvironmentalConditions(co2_ppm=1000.0)
        n_high_co2 = ciddor_refractive_index(self.REF_WAVELENGTH_UM, conditions_high_co2)
        assert n_high_co2 > (1.0 + self.REF_N_MINUS_1_CIDDOR)

        conditions_low_co2 = EnvironmentalConditions(co2_ppm=0.0)
        n_low_co2 = ciddor_refractive_index(self.REF_WAVELENGTH_UM, conditions_low_co2)
        assert n_low_co2 < (1.0 + self.REF_N_MINUS_1_CIDDOR)

    def test_ciddor_refractive_index_effect_of_humidity(self, set_test_backend):
        """
        Verifies that increasing humidity decreases the refractive index.
        """
        moist_conditions = EnvironmentalConditions(relative_humidity=0.8)
        dry_conditions = EnvironmentalConditions(relative_humidity=0.0)
        n_moist = ciddor_refractive_index(0.55, moist_conditions)
        n_dry = ciddor_refractive_index(0.55, dry_conditions)
        assert n_moist < n_dry

    def test_ciddor_refractive_index_effect_of_temperature(self, set_test_backend):
        """
        Verifies that increasing temperature decreases the refractive index.
        """
        cold_conditions = EnvironmentalConditions(temperature=0.0)
        hot_conditions = EnvironmentalConditions(temperature=30.0)
        n_cold = ciddor_refractive_index(self.REF_WAVELENGTH_UM, cold_conditions)
        n_hot = ciddor_refractive_index(self.REF_WAVELENGTH_UM, hot_conditions)
        assert n_cold > n_hot

    def test_ciddor_refractive_index_effect_of_pressure(self, set_test_backend):
        """
        Verifies that increasing pressure increases the refractive index.
        """
        low_p_conditions = EnvironmentalConditions(pressure=50000.0)
        high_p_conditions = EnvironmentalConditions(pressure=200000.0)
        n_low_p = ciddor_refractive_index(self.REF_WAVELENGTH_UM, low_p_conditions)
        n_high_p = ciddor_refractive_index(self.REF_WAVELENGTH_UM, high_p_conditions)
        assert n_low_p < n_high_p

    def test_ciddor_refractive_index_effect_of_wavelength(self, set_test_backend):
        """
        Verifies the dispersion of the model (shorter wavelength gives
        higher refractive index).
        """
        n_blue = ciddor_refractive_index(0.4, self.std_conditions)
        n_red = ciddor_refractive_index(0.7, self.std_conditions)
        assert n_blue > n_red

    def test_ciddor_refractive_index_zero_pressure(self, set_test_backend):
        """
        Tests that at zero pressure, the refractive index is 1 (vacuum).
        """
        conditions_vacuum = EnvironmentalConditions(pressure=0.0)
        n_vacuum = ciddor_refractive_index(self.REF_WAVELENGTH_UM, conditions_vacuum)
        assert_allclose(n_vacuum, 1.0, atol=1e-9)

    def test_ciddor_refractive_index_invalid_wavelength(self, set_test_backend):
        """
        Tests that a non-positive wavelength raises a ValueError.
        """
        with pytest.raises(ValueError, match="Wavelength must be positive."):
            ciddor_refractive_index(0.0, self.std_conditions)

    def test_ciddor_refractive_index_invalid_conditions_type(self, set_test_backend):
        """
        Tests that providing an invalid type for the `conditions` argument
        raises a TypeError.
        """
        with pytest.raises(TypeError, match="conditions must be an EnvironmentalConditions object."):
            ciddor_refractive_index(self.REF_WAVELENGTH_UM, "not_a_condition_object")