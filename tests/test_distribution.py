import pytest
import numpy as np
from optiland import distribution


@pytest.mark.parametrize('num_points', [10, 25, 106, 512])
def test_line_x(num_points):
    d = distribution.create_distribution('line_x')
    d.generate_points(num_points=num_points)

    assert np.allclose(d.dx, 2.0 / (num_points - 1))
    assert np.allclose(d.dy, 0.0)
    assert np.allclose(d.x, np.linspace(-1, 1, num_points))
    assert np.allclose(d.y, np.zeros(num_points))

    d = distribution.create_distribution('positive_line_x')
    d.generate_points(num_points=num_points)

    assert np.allclose(d.dx, 1.0 / (num_points - 1))
    assert np.allclose(d.dy, 0.0)
    assert np.allclose(d.x, np.linspace(0, 1, num_points))
    assert np.allclose(d.y, np.zeros(num_points))


@pytest.mark.parametrize('num_points', [9, 60, 111, 509])
def test_line_y(num_points):
    d = distribution.create_distribution('line_y')
    d.generate_points(num_points=num_points)

    assert np.allclose(d.dx, 0.0)
    assert np.allclose(d.dy, 2.0 / (num_points - 1))
    assert np.allclose(d.x, np.zeros(num_points))
    assert np.allclose(d.y, np.linspace(-1, 1, num_points))

    d = distribution.create_distribution('positive_line_y')
    d.generate_points(num_points=num_points)

    assert np.allclose(d.dx, 0.0)
    assert np.allclose(d.dy, 1.0 / (num_points - 1))
    assert np.allclose(d.x, np.zeros(num_points))
    assert np.allclose(d.y, np.linspace(0, 1, num_points))


@pytest.mark.parametrize('num_points', [8, 26, 154, 689])
def test_random(num_points):
    seed = 42
    d = distribution.RandomDistribution(seed=seed)
    d.generate_points(num_points=num_points)

    rng = np.random.default_rng(seed=seed)
    r = rng.uniform(size=num_points)
    theta = rng.uniform(0, 2*np.pi, size=num_points)

    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)

    assert np.allclose(d.x, x)
    assert np.allclose(d.y, y)


@pytest.mark.parametrize('num_rings', [3, 7, 15, 220])
def test_hexapolar(num_rings):
    d = distribution.create_distribution('hexapolar')
    d.generate_points(num_rings=num_rings)

    x = np.zeros(1)
    y = np.zeros(1)
    r = np.linspace(0, 1, num_rings + 1)

    for i in range(num_rings):
        num_theta = 6 * (i + 1)
        theta = np.linspace(0, 2 * np.pi, num_theta + 1)[:-1]
        x = np.concatenate([x, r[i + 1] * np.cos(theta)])
        y = np.concatenate([y, r[i + 1] * np.sin(theta)])

    assert np.allclose(d.x, x)
    assert np.allclose(d.y, y)


@pytest.mark.parametrize('num_points', [15, 56, 161, 621])
def test_cross(num_points):
    d = distribution.create_distribution('cross')
    d.generate_points(num_points=num_points)

    x1 = np.zeros(num_points)
    x2 = np.linspace(-1, 1, num_points)
    y1 = np.linspace(-1, 1, num_points)
    y2 = np.zeros(num_points)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    assert np.allclose(d.x, x)
    assert np.allclose(d.y, y)
