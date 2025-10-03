import pytest
from optiland.environment.environment import Environment


class DummyMaterial:
    name = "Dummy"
    def __repr__(self):
        return "<DummyMaterial>"

class DummyConditions:
    temperature = 25.0
    pressure = 1.0
    def __repr__(self):
        return "<DummyConditions>"

def test_environment_init_assigns_attributes(set_test_backend):
    medium = DummyMaterial()
    conditions = DummyConditions()
    env = Environment(medium=medium, conditions=conditions)
    assert env.medium is medium
    assert env.conditions is conditions

def test_environment_medium_attribute(set_test_backend):
    medium = DummyMaterial()
    env = Environment(medium=medium, conditions=DummyConditions())
    assert hasattr(env, "medium")
    assert env.medium.name == "Dummy"

def test_environment_conditions_attribute(set_test_backend):
    conditions = DummyConditions()
    env = Environment(medium=DummyMaterial(), conditions=conditions)
    assert hasattr(env, "conditions")
    assert env.conditions.temperature == 25.0
    assert env.conditions.pressure == 1.0

def test_environment_repr_of_attributes(set_test_backend):
    medium = DummyMaterial()
    conditions = DummyConditions()
    env = Environment(medium=medium, conditions=conditions)
    assert "<DummyMaterial>" in repr(env.medium)
    assert "<DummyConditions>" in repr(env.conditions)
