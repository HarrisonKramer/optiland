# tests/distribution/test_distribution_points.py
"""
Tests for the point generation functions in optiland.distribution.

This file verifies that the various distribution creation functions
(e.g., 'line_x', 'random', 'hexapolar') generate the correct set of
points for pupil sampling.
"""
import pytest
import matplotlib.pyplot as plt

import optiland.backend as be
from optiland import distribution
from ..utils import assert_allclose


@pytest.mark.parametrize("num_points", [10, 25, 106, 512])
def test_line_x(set_test_backend, num_points):
    """
    Tests the 'line_x' and 'positive_line_x' distributions, which should
    generate points along the x-axis.
    """
    # Test 'line_x' (from -1 to 1)
    d_full = distribution.create_distribution("line_x")
    d_full.generate_points(num_points=num_points)
    assert_allclose(d_full.x, be.linspace(-1, 1, num_points))
    assert_allclose(d_full.y, be.zeros(num_points))

    # Test 'positive_line_x' (from 0 to 1)
    d_pos = distribution.create_distribution("positive_line_x")
    d_pos.generate_points(num_points=num_points)
    assert_allclose(d_pos.x, be.linspace(0, 1, num_points))
    assert_allclose(d_pos.y, be.zeros(num_points))


@pytest.mark.parametrize("num_points", [9, 60, 111, 509])
def test_line_y(set_test_backend, num_points):
    """
    Tests the 'line_y' and 'positive_line_y' distributions, which should
    generate points along the y-axis.
    """
    # Test 'line_y' (from -1 to 1)
    d_full = distribution.create_distribution("line_y")
    d_full.generate_points(num_points=num_points)
    assert_allclose(d_full.x, be.zeros(num_points))
    assert_allclose(d_full.y, be.linspace(-1, 1, num_points))

    # Test 'positive_line_y' (from 0 to 1)
    d_pos = distribution.create_distribution("positive_line_y")
    d_pos.generate_points(num_points=num_points)
    assert_allclose(d_pos.x, be.zeros(num_points))
    assert_allclose(d_pos.y, be.linspace(0, 1, num_points))


@pytest.mark.parametrize("num_points", [8, 26, 154, 689])
def test_random(set_test_backend, num_points):
    """
    Tests the 'random' distribution by comparing its output against a
    known random seed.
    """
    seed = 42
    d = distribution.RandomDistribution(seed=seed)
    d.generate_points(num_points=num_points)

    # Re-generate the same random points for comparison
    rng = be.default_rng(seed=seed)
    r = be.random_uniform(size=num_points, generator=rng)
    theta = be.random_uniform(0, 2 * be.pi, size=num_points, generator=rng)
    x = be.sqrt(r) * be.cos(theta)
    y = be.sqrt(r) * be.sin(theta)

    assert_allclose(d.x, x)
    assert_allclose(d.y, y)


@pytest.mark.parametrize("num_rings", [3, 7, 15, 22])
def test_hexapolar(set_test_backend, num_rings):
    """
    Tests the 'hexapolar' distribution, which generates points in
    concentric hexagonal rings.
    """
    d = distribution.create_distribution("hexapolar")
    d.generate_points(num_rings=num_rings)

    # Manually construct the expected hexapolar pattern for validation
    x = be.zeros(1)
    y = be.zeros(1)
    r = be.linspace(0, 1, num_rings + 1)
    for i in range(num_rings):
        num_theta = 6 * (i + 1)
        theta = be.linspace(0, 2 * be.pi, num_theta + 1)[:-1]
        x = be.concatenate([x, r[i + 1] * be.cos(theta)])
        y = be.concatenate([y, r[i + 1] * be.sin(theta)])

    assert_allclose(d.x, x)
    assert_allclose(d.y, y)


@pytest.mark.parametrize("num_points", [15, 56, 161, 621])
def test_cross(set_test_backend, num_points):
    """
    Tests the 'cross' distribution, which generates points along the
    x and y axes.
    """
    d = distribution.create_distribution("cross")
    d.generate_points(num_points=num_points)

    # Manually construct the expected cross pattern for validation
    line_y = be.linspace(-1, 1, num_points)
    line_x = be.linspace(-1, 1, num_points)
    zeros = be.zeros(num_points)

    # Handle the duplicated origin point for odd numbers of points
    if num_points % 2 == 1:
        mid_idx = num_points // 2
        line_x_no_origin = be.concatenate((line_x[:mid_idx], line_x[mid_idx + 1 :]))
        zeros_no_origin = be.concatenate((zeros[:mid_idx], zeros[mid_idx + 1 :]))
    else:
        line_x_no_origin = line_x
        zeros_no_origin = zeros

    expected_x = be.concatenate((zeros, line_x_no_origin))
    expected_y = be.concatenate((line_y, zeros_no_origin))

    # Sort both arrays to ensure a consistent order for comparison
    sort_indices_d = be.argsort(d.x)
    sort_indices_expected = be.argsort(expected_x)

    assert_allclose(d.x[sort_indices_d], expected_x[sort_indices_expected])
    assert_allclose(d.y[sort_indices_d], expected_y[sort_indices_expected])


def test_view_distribution(set_test_backend):
    """
    Tests that the `view` method for plotting a distribution runs without
    error and returns a valid matplotlib Figure and Axes.
    """
    d = distribution.create_distribution("random")
    d.generate_points(num_points=10)
    fig, ax = d.view()
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_invalid_distribution_error(set_test_backend):
    """
    Tests that requesting an unknown distribution type raises a ValueError.
    """
    with pytest.raises(ValueError):
        distribution.create_distribution(distribution_type="invalid")


@pytest.mark.parametrize("num_points", [10, 25, 50, 100])
def test_uniform_distribution(set_test_backend, num_points):
    """
    Tests the 'uniform' (or 'grid') distribution, which generates points on
    a square grid within the unit circle.
    """
    d = distribution.create_distribution("uniform")
    d.generate_points(num_points=num_points)

    # Manually construct the expected grid for validation
    x_line = be.linspace(-1, 1, num_points)
    x, y = be.meshgrid(x_line, x_line)
    r2 = x**2 + y**2
    x_expected = x[r2 <= 1]
    y_expected = y[r2 <= 1]

    assert be.all(d.x >= -1.0) and be.all(d.x <= 1.0)
    assert be.all(d.y >= -1.0) and be.all(d.y <= 1.0)
    assert_allclose(d.x, x_expected)
    assert_allclose(d.y, y_expected)


def test_gaussian_quad_distribution(set_test_backend):
    """
    Tests the 'gaussian_quadrature' distribution against known values for
    both symmetric and asymmetric patterns.
    """
    # Test symmetric case
    d_sym = distribution.GaussianQuadrature(is_symmetric=True)
    d_sym.generate_points(num_rings=3)
    assert_allclose(d_sym.x, be.array([0.33571, 0.70711, 0.94196]))

    # Test asymmetric case
    d_asym = distribution.GaussianQuadrature(is_symmetric=False)
    d_asym.generate_points(num_rings=2)
    assert_allclose(d_asym.x[3], 0.444035)
    assert_allclose(d_asym.y[3], -0.769091)


def test_gaussian_quad_distribution_errors(set_test_backend):
    """
    Tests that the GaussianQuadrature distribution raises ValueErrors for
    invalid input (e.g., zero or negative number of rings).
    """
    with pytest.raises(ValueError):
        distribution.GaussianQuadrature().generate_points(num_rings=0)
    with pytest.raises(ValueError):
        distribution.GaussianQuadrature().get_weights(num_rings=0)


def test_gaussian_quad_weights(set_test_backend):
    """
    Tests that the weights for the Gaussian quadrature are calculated
    correctly against known values.
    """
    d = distribution.GaussianQuadrature(is_symmetric=True)
    weights = d.get_weights(num_rings=3)
    assert_allclose(weights, be.array([0.13889, 0.22222, 0.13889]) * 6.0, atol=1e-4)