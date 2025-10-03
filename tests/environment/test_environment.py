"""Tests for the environment feature."""
import pytest

from optiland import backend as be
from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.manager import environment_manager
from optiland.materials.air import Air
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from tests.utils import assert_allclose


def test_air_refractive_index_is_one_in_nominal_case(set_test_backend):
    """Test that the refractive index of Air is 1.0 in the nominal case."""
    # Reset to default environment to ensure nominal conditions
    environment_manager.reset_to_default()

    # Get the primary wavelength from a default Optic instance
    optic = Optic()
    optic.add_wavelength(value=0.55, is_primary=True)
    wavelength = optic.primary_wavelength

    # Get the refractive index of the environment medium
    air_index = environment_manager.get_environment().medium.n(wavelength)

    # The refractive index of air relative to itself should be exactly 1.0
    assert_allclose(air_index, 1.0)


def test_system_with_pressurized_air(set_test_backend):
    """Test a system with pressurized air as the immersion medium."""
    # Define pressurized conditions
    pressurized_conditions = EnvironmentalConditions(
        temperature=20.0, pressure=2 * 101325.0
    )
    from optiland.environment.environment import Environment
    environment_manager.set_environment(
        Environment(
            medium=Air(conditions=pressurized_conditions),
            conditions=pressurized_conditions,
        )
    )

    # Get the primary wavelength from a default Optic instance
    optic = Optic()
    optic.add_wavelength(value=0.55, is_primary=True)
    wavelength = optic.primary_wavelength

    # Get the refractive index of the environment medium
    # We must call _calculate_absolute_n here because the public n() method
    # will return 1.0 for the environment medium itself.
    pressurized_air_index = environment_manager.get_environment().medium._calculate_absolute_n(
        wavelength
    )

    # The refractive index should be greater than 1.0
    assert pressurized_air_index > 1.0

    # Reset to default to avoid affecting other tests
    environment_manager.reset_to_default()


def test_system_with_water_immersion(set_test_backend):
    """Test a system with water as the immersion medium."""
    # Define water as the immersion medium
    from optiland.environment.environment import Environment
    water = IdealMaterial(n=1.333)
    conditions = EnvironmentalConditions()
    environment_manager.set_environment(Environment(medium=water, conditions=conditions))

    # Get the primary wavelength from a default Optic instance
    optic = Optic()
    optic.add_wavelength(value=0.55, is_primary=True)
    wavelength = optic.primary_wavelength

    # Get the refractive index of the environment medium
    water_index = environment_manager.get_environment().medium.n(wavelength)

    # The refractive index should be 1.333
    assert_allclose(water_index, 1.333)

    # Reset to default to avoid affecting other tests
    environment_manager.reset_to_default()