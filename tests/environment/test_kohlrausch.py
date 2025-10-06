# tests/environment/test_kohlrausch.py
"""
Tests for the Kohlrausch (1968) dispersion formula for the refractive index of air.
"""
from __future__ import annotations

from math import isclose

import pytest

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.models.kohlrausch import (
    ALPHA_T,
    T_REF_C,
    kohlrausch_refractive_index,
)


class TestKohlrauschRefractiveIndex:
    """
    Tests the Kohlrausch (1968) model for calculating the refractive index of air.
    It verifies the calculation under various environmental conditions and checks
    input validation.
    """

    STD_TEMP_C = 15.0
    STD_PRES_PA = 101325.0
    REF_WAVELENGTH_UM = 0.55
    # Reference value calculated from the authoritative formula for standard conditions.
    REF_N_KOHLRAUSCH = 1.000277328

    def setup_method(self):
        """
        Sets up a standard set of environmental conditions based on the
        Kohlrausch model's reference. Kohlrausch ignores humidity and CO2.
        """
        self.std_conditions = EnvironmentalConditions(
            temperature=self.STD_TEMP_C,
            pressure=self.STD_PRES_PA,
        )

    def test_kohlrausch_refractive_index_standard_conditions(self):
        """
        Tests the Kohlrausch model with standard conditions and compares against
        a known reference value.
        """
        n_kohlrausch = kohlrausch_refractive_index(
            self.REF_WAVELENGTH_UM, self.std_conditions
        )
        assert isclose(n_kohlrausch, self.REF_N_KOHLRAUSCH, rel_tol=1e-8)

    def test_kohlrausch_refractive_index_temperature_effect(self):
        """
        Verifies that increasing temperature decreases the refractive index
        (due to lower air density).
        """
        cold_conditions = EnvironmentalConditions(temperature=0.0, pressure=self.STD_PRES_PA)
        hot_conditions = EnvironmentalConditions(temperature=30.0, pressure=self.STD_PRES_PA)
        n_cold = kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, cold_conditions)
        n_hot = kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, hot_conditions)
        assert n_cold > n_hot

    def test_kohlrausch_refractive_index_pressure_effect(self):
        """
        Verifies that increasing pressure increases the refractive index
        (due to higher air density).
        """
        low_p_conditions = EnvironmentalConditions(temperature=self.STD_TEMP_C, pressure=50000.0)
        high_p_conditions = EnvironmentalConditions(temperature=self.STD_TEMP_C, pressure=200000.0)
        n_low_p = kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, low_p_conditions)
        n_high_p = kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, high_p_conditions)
        assert n_low_p < n_high_p

    def test_kohlrausch_refractive_index_wavelength_effect(self):
        """
        Verifies the dispersion of the model (shorter wavelength gives
        higher refractive index).
        """
        n_blue = kohlrausch_refractive_index(0.4, self.std_conditions)
        n_red = kohlrausch_refractive_index(0.7, self.std_conditions)
        assert n_blue > n_red

    def test_kohlrausch_refractive_index_zero_pressure(self):
        """
        Tests that at zero pressure, the refractive index is 1 (vacuum).
        """
        conditions_vacuum = EnvironmentalConditions(pressure=0.0)
        n_vacuum = kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, conditions_vacuum)
        assert isclose(n_vacuum, 1.0, abs_tol=1e-9)

    def test_kohlrausch_refractive_index_invalid_wavelength_zero(self):
        """
        Tests that a zero wavelength input raises a ValueError.
        """
        with pytest.raises(ValueError, match="Wavelength must be non-zero."):
            kohlrausch_refractive_index(0.0, self.std_conditions)

    def test_kohlrausch_refractive_index_invalid_conditions_type(self):
        """
        Tests that providing an invalid type for the `conditions` argument
        raises an AttributeError.
        """
        with pytest.raises(AttributeError):
            kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, "not_a_condition_object")

    def test_kohlrausch_refractive_index_extreme_low_temperature(self):
        """
        Tests that a temperature leading to a non-positive denominator in the
        formula raises a ValueError.
        """
        critical_temp = T_REF_C - (1.0 / ALPHA_T)
        conditions_extreme_cold = EnvironmentalConditions(temperature=critical_temp - 0.1)
        with pytest.raises(ValueError, match="Invalid temperature.*non-positive denominator."):
            kohlrausch_refractive_index(self.REF_WAVELENGTH_UM, conditions_extreme_cold)