
import pytest

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry
from tests.utils import assert_allclose


def test_nurbs_geometry_init(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(cs, nurbs_norm_x=20, nurbs_norm_y=20, n_points_u=10, n_points_v=10)
    geo.fit_surface()
    assert geo is not None


def test_nurbs_geometry_sag(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs, radius=100, conic=-1, nurbs_norm_x=20, nurbs_norm_y=20, n_points_u=10, n_points_v=10
    )
    geo.fit_surface()
    sag = geo.sag(
        x=be.asarray([0, 10], dtype=be.float64), y=be.asarray([0, 0], dtype=be.float64)
    )
    assert_allclose(sag[0], 0.0, atol=1e-4)
    assert_allclose(sag[1], 0.5, atol=1e-4)


class MockRays:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_nurbs_geometry_normal(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs, radius=100, conic=-1, nurbs_norm_x=20, nurbs_norm_y=20, n_points_u=10, n_points_v=10
    )
    geo.fit_surface()
    rays = MockRays(
        x=be.asarray([0], dtype=be.float64), y=be.asarray([0], dtype=be.float64)
    )
    nx, ny, nz = geo.surface_normal(rays)
    assert_allclose(nx, 0.0, atol=1e-4)
    assert_allclose(ny, 0.0, atol=1e-4)
    assert_allclose(nz, 1.0, atol=1e-4)


def test_nurbs_get_value(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs, radius=100, conic=-1, nurbs_norm_x=20, nurbs_norm_y=20, n_points_u=10, n_points_v=10
    )
    geo.fit_surface()
    # Test a point that should be on the surface
    val = geo.get_value(
        u=be.asarray([0.5], dtype=be.float64), v=be.asarray([0.5], dtype=be.float64)
    )
    assert val.shape == (3, 1)


def test_nurbs_get_derivative(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs, radius=100, conic=-1, nurbs_norm_x=20, nurbs_norm_y=20, n_points_u=10, n_points_v=10
    )
    geo.fit_surface()
    # Test derivative at a point
    derivative = geo.get_derivative(
        u=be.asarray([0.5], dtype=be.float64),
        v=be.asarray([0.5], dtype=be.float64),
        order_u=1,
        order_v=0,
    )
    assert derivative.shape == (3, 1)


class MockRaysDistance:
    def __init__(self, x, y, z, L, M, N):
        self.x = x
        self.y = y
        self.z = z
        self.L = L
        self.M = M
        self.N = N


def test_nurbs_distance(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs, radius=100, conic=-1, nurbs_norm_x=20, nurbs_norm_y=20, n_points_u=10, n_points_v=10
    )
    geo.fit_surface()
    # Test distance to a ray
    rays = MockRaysDistance(
        x=be.asarray([0.0], dtype=be.float64),
        y=be.asarray([0.0], dtype=be.float64),
        z=be.asarray([-10.0], dtype=be.float64),
        L=be.asarray([0.0], dtype=be.float64),
        M=be.asarray([0.0], dtype=be.float64),
        N=be.asarray([1.0], dtype=be.float64),
    )
    distance = geo.distance(rays)
    assert_allclose(distance, 10, atol=1e-4)

@pytest.fixture
def sample_control_points():
    """Provides a simple 3x3 grid of 3D control points as a Python list."""
    dim, n, m = 3, 3, 3
    points = [[[0.0 for _ in range(m)] for _ in range(n)] for _ in range(dim)]
    for i in range(n):
        for j in range(m):
            points[0][i][j] = float(i)
            points[1][i][j] = float(j)
            points[2][i][j] = 0.0
    return points

@pytest.mark.parametrize('backend', be.list_available_backends())
def test_nurbs_geometry_init_fitted(backend, set_test_backend):
    """Tests the initialization of a fitted NurbsGeometry."""
    be.set_backend(backend)
    cs = CoordinateSystem()

    # Initialize without control points to trigger fitting logic
    geom = NurbsGeometry(cs, n_points_u=5, n_points_v=5)

    assert geom.is_fitted
    assert geom.P is None
    assert geom.W is None
    assert geom.ndim == 3
    assert geom.P_size_u == 6  # n_points_u + 1
    assert geom.P_size_v == 6  # n_points_v + 1

@pytest.mark.parametrize('backend', be.list_available_backends())
def test_nurbs_geometry_init_bezier(backend, set_test_backend, sample_control_points):
    """Tests the Polynomial Bezier surface initialization of NurbsGeometry."""
    be.set_backend(backend)
    cs = CoordinateSystem()
    control_points = be.asarray(sample_control_points)

    geom = NurbsGeometry(cs, control_points=control_points)

    assert not geom.is_fitted
    assert geom.surface_type == "Bezier"
    assert be.allclose(geom.P, control_points)

    n = be.shape(control_points)[1] - 1
    m = be.shape(control_points)[2] - 1

    assert geom.p == n
    assert geom.q == m
    assert geom.U is not None
    assert geom.V is not None
    assert be.allclose(geom.W, be.ones((n + 1, m + 1)))
