from unittest.mock import patch

import optiland.backend as be
import pytest

from optiland import distribution


@pytest.mark.parametrize("num_points", [10, 25, 106, 512])
def test_line_x(num_points):
    d = distribution.create_distribution("line_x")
    d.generate_points(num_points=num_points)

    assert be.allclose(d.x, be.linspace(-1, 1, num_points))
    assert be.allclose(d.y, be.zeros(num_points))

    d = distribution.create_distribution("positive_line_x")
    d.generate_points(num_points=num_points)

    assert be.allclose(d.x, be.linspace(0, 1, num_points))
    assert be.allclose(d.y, be.zeros(num_points))


@pytest.mark.parametrize("num_points", [9, 60, 111, 509])
def test_line_y(num_points):
    d = distribution.create_distribution("line_y")
    d.generate_points(num_points=num_points)

    assert be.allclose(d.x, be.zeros(num_points))
    assert be.allclose(d.y, be.linspace(-1, 1, num_points))

    d = distribution.create_distribution("positive_line_y")
    d.generate_points(num_points=num_points)

    assert be.allclose(d.x, be.zeros(num_points))
    assert be.allclose(d.y, be.linspace(0, 1, num_points))


@pytest.mark.parametrize("num_points", [8, 26, 154, 689])
def test_random(num_points):
    seed = 42
    d = distribution.RandomDistribution(seed=seed)
    d.generate_points(num_points=num_points)

    rng = be.random.default_rng(seed=seed)
    r = rng.uniform(size=num_points)
    theta = rng.uniform(0, 2 * be.pi, size=num_points)

    x = be.sqrt(r) * be.cos(theta)
    y = be.sqrt(r) * be.sin(theta)

    assert be.allclose(d.x, x)
    assert be.allclose(d.y, y)


@pytest.mark.parametrize("num_rings", [3, 7, 15, 220])
def test_hexapolar(num_rings):
    d = distribution.create_distribution("hexapolar")
    d.generate_points(num_rings=num_rings)

    x = be.zeros(1)
    y = be.zeros(1)
    r = be.linspace(0, 1, num_rings + 1)

    for i in range(num_rings):
        num_theta = 6 * (i + 1)
        theta = be.linspace(0, 2 * be.pi, num_theta + 1)[:-1]
        x = be.concatenate([x, r[i + 1] * be.cos(theta)])
        y = be.concatenate([y, r[i + 1] * be.sin(theta)])

    assert be.allclose(d.x, x)
    assert be.allclose(d.y, y)


@pytest.mark.parametrize("num_points", [15, 56, 161, 621])
def test_cross(num_points):
    d = distribution.create_distribution("cross")
    d.generate_points(num_points=num_points)

    x1 = be.zeros(num_points)
    x2 = be.linspace(-1, 1, num_points)
    y1 = be.linspace(-1, 1, num_points)
    y2 = be.zeros(num_points)
    x = be.concatenate((x1, x2))
    y = be.concatenate((y1, y2))

    assert be.allclose(d.x, x)
    assert be.allclose(d.y, y)


@patch("matplotlib.pyplot.show")
def test_view_distribution(mock_show):
    d = distribution.create_distribution("random")
    d.generate_points(num_points=10)
    d.view()
    mock_show.assert_called_once()


def test_invalid_distribution_error():
    with pytest.raises(ValueError):
        distribution.create_distribution(distribution_type="invalid")


@pytest.mark.parametrize("num_points", [10, 25, 50, 100])
def test_uniform_distribution(num_points):
    d = distribution.create_distribution("uniform")
    d.generate_points(num_points=num_points)

    x = be.linspace(-1, 1, num_points)
    x, y = be.meshgrid(x, x)
    r2 = x**2 + y**2
    x = x[r2 <= 1]
    y = y[r2 <= 1]

    assert be.all(d.x >= -1.0) and be.all(d.x <= 1.0)
    assert be.all(d.y >= -1.0) and be.all(d.y <= 1.0)
    assert be.allclose(d.x, x)
    assert be.allclose(d.y, y)


def test_gaussian_quad_distribution():
    # 1 ring - symmetric
    d = distribution.GaussianQuadrature(is_symmetric=True)
    d.generate_points(num_rings=1)
    assert be.allclose(d.x, be.array(0.70711))
    assert be.allclose(d.y, be.array(0.0))

    # 2 rings - symmetric
    d = distribution.GaussianQuadrature(is_symmetric=True)
    d.generate_points(num_rings=2)
    assert be.allclose(d.x, be.array([0.4597, 0.88807]))
    assert be.allclose(d.y, be.array([0.0, 0.0]))

    # 3 rings - symmetric
    d = distribution.GaussianQuadrature(is_symmetric=True)
    d.generate_points(num_rings=3)
    assert be.allclose(d.x, be.array([0.33571, 0.70711, 0.94196]))
    assert be.allclose(d.y, be.array([0.0, 0.0, 0.0]))

    # 4 rings - symmetric
    d = distribution.GaussianQuadrature(is_symmetric=True)
    d.generate_points(num_rings=4)
    assert be.allclose(d.x, be.array([0.2635, 0.57446, 0.81853, 0.96466]))
    assert be.allclose(d.y, be.array([0.0, 0.0, 0.0, 0.0]))

    # 5 rings - symmetric
    d = distribution.GaussianQuadrature(is_symmetric=True)
    d.generate_points(num_rings=5)
    assert be.allclose(d.x, be.array([0.21659, 0.48038, 0.70711, 0.87706, 0.97626]))
    assert be.allclose(d.y, be.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    # 6 rings - symmetric
    d = distribution.GaussianQuadrature(is_symmetric=True)
    d.generate_points(num_rings=6)
    assert be.allclose(
        d.x,
        be.array([0.18375, 0.41158, 0.617, 0.78696, 0.91138, 0.983]),
    )
    assert be.allclose(d.y, be.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # 1 ring - asymmetric
    d = distribution.GaussianQuadrature(is_symmetric=False)
    d.generate_points(num_rings=1)
    assert be.allclose(
        d.x,
        be.array([0.35355500073276686, 0.70711, 0.35355500073276686]),
    )
    assert be.allclose(d.y, be.array([-0.6123752228469513, 0.0, 0.6123752228469513]))

    # 2 rings - asymmetric
    d = distribution.GaussianQuadrature(is_symmetric=False)
    d.generate_points(num_rings=2)
    assert be.allclose(
        d.x,
        be.array(
            [
                0.22985000047637977,
                0.4597,
                0.22985000047637977,
                0.4440350009202928,
                0.88807,
                0.4440350009202928,
            ],
        ),
    )
    assert be.allclose(
        d.y,
        be.array(
            [
                -0.39811187784466845,
                0.0,
                0.39811187784466845,
                -0.7690911798075152,
                0.0,
                0.7690911798075152,
            ],
        ),
    )

    # 3 rings - asymmetric
    d = distribution.GaussianQuadrature(is_symmetric=False)
    d.generate_points(num_rings=3)
    assert be.allclose(
        d.x,
        be.array(
            [
                0.16785500034789091,
                0.33571,
                0.16785500034789091,
                0.35355500073276686,
                0.70711,
                0.35355500073276686,
                0.4709800009761381,
                0.94196,
                0.4709800009761381,
            ],
        ),
    )
    assert be.allclose(
        d.y,
        be.array(
            [
                -0.290733388103619,
                0.0,
                0.290733388103619,
                -0.6123752228469513,
                0.0,
                0.6123752228469513,
                -0.8157612887852163,
                0.0,
                0.8157612887852163,
            ],
        ),
    )

    # 4 rings - asymmetric
    d = distribution.GaussianQuadrature(is_symmetric=False)
    d.generate_points(num_rings=4)
    assert be.allclose(
        d.x,
        be.array(
            [
                0.13175000027306086,
                0.2635,
                0.13175000027306086,
                0.28723000059530374,
                0.57446,
                0.28723000059530374,
                0.4092650008482296,
                0.81853,
                0.4092650008482296,
                0.4823300009996618,
                0.96466,
                0.4823300009996618,
            ],
        ),
    )
    assert be.allclose(
        d.y,
        be.array(
            [
                -0.22819769373954782,
                0.0,
                0.22819769373954782,
                -0.4974969531143098,
                0.0,
                0.4974969531143098,
                -0.708867773269951,
                0.0,
                0.708867773269951,
                -0.8354200654375415,
                0.0,
                0.8354200654375415,
            ],
        ),
    )

    # 5 rings - asymmetric
    d = distribution.GaussianQuadrature(is_symmetric=False)
    d.generate_points(num_rings=5)
    assert be.allclose(
        d.x,
        be.array(
            [
                0.10829500022444877,
                0.21659,
                0.10829500022444877,
                0.2401900004978101,
                0.48038,
                0.2401900004978101,
                0.35355500073276686,
                0.70711,
                0.35355500073276686,
                0.4385300009088833,
                0.87706,
                0.4385300009088833,
                0.4881300010116827,
                0.97626,
                0.4881300010116827,
            ],
        ),
    )
    assert be.allclose(
        d.y,
        be.array(
            [
                -0.187572442076086,
                0.0,
                0.187572442076086,
                -0.41602128318255777,
                0.0,
                0.41602128318255777,
                -0.6123752228469513,
                0.0,
                0.6123752228469513,
                -0.7595562401184357,
                0.0,
                0.7595562401184357,
                -0.8454659601145008,
                0.0,
                0.8454659601145008,
            ],
        ),
    )

    # 6 rings - asymmetric
    d = distribution.GaussianQuadrature(is_symmetric=False)
    d.generate_points(num_rings=6)
    assert be.allclose(
        d.x,
        be.array(
            [
                0.0918750001904172,
                0.18375,
                0.0918750001904172,
                0.2057900004265138,
                0.41158,
                0.2057900004265138,
                0.30850000063938726,
                0.617,
                0.30850000063938726,
                0.3934800008155141,
                0.78696,
                0.3934800008155141,
                0.45569000094444856,
                0.91138,
                0.45569000094444856,
                0.4915000010186672,
                0.983,
                0.4915000010186672,
            ],
        ),
    )
    assert be.allclose(
        d.y,
        be.array(
            [
                -0.15913216783545317,
                0.0,
                0.15913216783545317,
                -0.3564387354433514,
                0.0,
                0.3564387354433514,
                -0.5343376737658482,
                0.0,
                0.5343376737658482,
                -0.6815273512913645,
                0.0,
                0.6815273512913645,
                -0.7892782319557841,
                0.0,
                0.7892782319557841,
                -0.8513029713319753,
                0.0,
                0.8513029713319753,
            ],
        ),
    )


def test_gaussian_quad_distribution_errors():
    with pytest.raises(ValueError):
        d = distribution.GaussianQuadrature(is_symmetric=True)
        d.generate_points(num_rings=0)

    with pytest.raises(ValueError):
        d = distribution.GaussianQuadrature(is_symmetric=True)
        d.generate_points(num_rings=-1)

    with pytest.raises(ValueError):
        d = distribution.GaussianQuadrature(is_symmetric=False)
        d.generate_points(num_rings=0)

    with pytest.raises(ValueError):
        d = distribution.GaussianQuadrature(is_symmetric=False)
        d.generate_points(num_rings=-1)

    with pytest.raises(ValueError):
        d = distribution.GaussianQuadrature(is_symmetric=False)
        d.get_weights(num_rings=10)

    with pytest.raises(ValueError):
        d = distribution.GaussianQuadrature(is_symmetric=False)
        d.get_weights(num_rings=-1)

    with pytest.raises(ValueError):
        d = distribution.GaussianQuadrature(is_symmetric=False)
        d.get_weights(num_rings=0)


def test_gaussian_quad_weights():
    scale = [6.0, 2.0]
    for k, is_symmetric in enumerate([True, False]):
        # 1 ring
        d = distribution.GaussianQuadrature(is_symmetric=is_symmetric)
        weights = d.get_weights(num_rings=1)
        assert be.allclose(weights / scale[k], be.array([0.5]))

        # 2 rings
        d = distribution.GaussianQuadrature(is_symmetric=is_symmetric)
        weights = d.get_weights(num_rings=2)
        assert be.allclose(weights / scale[k], be.array([0.25, 0.25]))

        # 3 rings
        d = distribution.GaussianQuadrature(is_symmetric=is_symmetric)
        weights = d.get_weights(num_rings=3)
        assert be.allclose(weights / scale[k], be.array([0.13889, 0.22222, 0.13889]))

        # 4 rings
        d = distribution.GaussianQuadrature(is_symmetric=is_symmetric)
        weights = d.get_weights(num_rings=4)
        assert be.allclose(
            weights / scale[k],
            be.array([0.08696, 0.16304, 0.16304, 0.08696]),
        )

        # 5 rings
        d = distribution.GaussianQuadrature(is_symmetric=is_symmetric)
        weights = d.get_weights(num_rings=5)
        assert be.allclose(
            weights / scale[k],
            be.array([0.059231, 0.11966, 0.14222, 0.11966, 0.059231]),
        )

        # 6 rings
        d = distribution.GaussianQuadrature(is_symmetric=is_symmetric)
        weights = d.get_weights(num_rings=6)
        assert be.allclose(
            weights / scale[k],
            be.array([0.04283, 0.09019, 0.11698, 0.11698, 0.09019, 0.04283]),
        )
