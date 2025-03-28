import numpy as np
import pytest

from optiland.samples.objectives import TessarLens
from optiland.tolerancing.perturbation import (
    DistributionSampler,
    Perturbation,
    RangeSampler,
    ScalarSampler,
)


def test_scalar_sampler():
    sampler = ScalarSampler(5)
    assert sampler.sample() == 5
    assert sampler.size == 1


def test_range_sampler():
    sampler = RangeSampler(0, 10, 5)
    expected_values = np.linspace(0, 10, 5)
    for expected in expected_values:
        assert sampler.sample() == expected
    # Test looping over values
    assert sampler.sample() == expected_values[0]


def test_range_cycle_twice():
    sampler = RangeSampler(0, 10, 5)
    expected_values = np.linspace(0, 10, 5)
    for expected in expected_values:
        assert sampler.sample() == expected
    for expected in expected_values:
        assert sampler.sample() == expected


def test_distribution_sampler_normal():
    sampler = DistributionSampler("normal", seed=42, loc=0, scale=1)
    value = sampler.sample()
    np.random.seed(42)
    expected_value = np.random.normal(0, 1)
    assert np.isclose(value, expected_value)


def test_distribution_sampler_uniform():
    sampler = DistributionSampler("uniform", seed=42, low=0, high=1)
    value = sampler.sample()
    np.random.seed(42)
    expected_value = np.random.uniform(0, 1)
    assert np.isclose(value, expected_value)


def test_distribution_sampler_unknown():
    with pytest.raises(ValueError):
        DistributionSampler("unknown").sample()


def test_perturbation_apply():
    optic = TessarLens()
    sampler = ScalarSampler(1234)
    perturbation = Perturbation(optic, "radius", sampler, surface_number=1)
    perturbation.apply()
    assert perturbation.value == 1234
    assert perturbation.variable.value == 1234


def test_range_sampler_reset():
    sampler = RangeSampler(0, 10, 5)
    expected_values = np.linspace(0, 10, 5)
    for expected in expected_values:
        assert sampler.sample() == expected
    # Test looping over values
    assert sampler.sample() == expected_values[0]
    assert sampler.sample() == expected_values[1]


def test_distribution_sampler_seed():
    sampler1 = DistributionSampler("normal", seed=42, loc=0, scale=1)
    value1 = sampler1.sample()
    sampler2 = DistributionSampler("normal", seed=42, loc=0, scale=1)
    value2 = sampler2.sample()
    assert np.isclose(value1, value2)
