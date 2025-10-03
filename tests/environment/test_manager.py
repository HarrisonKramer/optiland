from __future__ import annotations

from optiland.environment.conditions import EnvironmentalConditions
from optiland.environment.environment import Environment
from optiland.environment.manager import EnvironmentManager, environment_manager
from optiland.materials.air import Air


def test_singleton_instance(set_test_backend):
    mgr1 = EnvironmentManager()
    mgr2 = EnvironmentManager()
    assert mgr1 is mgr2
    assert mgr1 is environment_manager


def test_default_environment_is_air(set_test_backend):
    mgr = EnvironmentManager()
    env = mgr.get_environment()
    assert isinstance(env, Environment)
    assert isinstance(env.medium, Air)
    assert isinstance(env.conditions, EnvironmentalConditions)


def test_set_and_get_environment(set_test_backend):
    mgr = EnvironmentManager()
    new_conditions = EnvironmentalConditions(temperature=300, pressure=1.05)
    new_air = Air(conditions=new_conditions)
    new_env = Environment(medium=new_air, conditions=new_conditions)
    mgr.set_environment(new_env)
    assert mgr.get_environment() is new_env


def test_reset_to_default_restores_air(set_test_backend):
    mgr = EnvironmentManager()
    # Change environment
    new_conditions = EnvironmentalConditions(temperature=290, pressure=0.9)
    new_air = Air(conditions=new_conditions)
    new_env = Environment(medium=new_air, conditions=new_conditions)
    mgr.set_environment(new_env)
    # Reset and check
    mgr.reset_to_default()
    env = mgr.get_environment()
    assert isinstance(env.medium, Air)
    assert isinstance(env.conditions, EnvironmentalConditions)
    # Should be default values
    assert env.conditions.temperature == EnvironmentalConditions().temperature
    assert env.conditions.pressure == EnvironmentalConditions().pressure
