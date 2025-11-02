"""Unit tests for the nurbs_fitting module."""
import pytest
import optiland.backend as be
from optiland.geometries.nurbs import nurbs_fitting

@pytest.fixture
def sample_points_curve():
    """Provides a simple list of 2D points for curve fitting tests."""
    return [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

@pytest.fixture
def sample_points_surface():
    """Provides a simple 4x4 grid of 3D points for surface fitting tests."""
    points = []
    for i in range(4):
        for j in range(4):
            points.append([float(i), float(j), 0.0])
    return points

@pytest.mark.parametrize('backend', be.list_available_backends())
def test_compute_params_curve(backend, sample_points_curve):
    """Tests the compute_params_curve function."""
    be.set_backend(backend)

    # Test with default chord length parametrization
    params = nurbs_fitting.compute_params_curve(sample_points_curve)
    assert len(params) == len(sample_points_curve)
    assert isinstance(params, list)
    assert all(isinstance(p, float) for p in params)
    assert params[0] == 0.0
    assert params[-1] == 1.0

    # Test with centripetal parametrization
    params_centripetal = nurbs_fitting.compute_params_curve(sample_points_curve, centripetal=True)
    assert len(params_centripetal) == len(sample_points_curve)
    assert params != params_centripetal  # Expect different values

    # Test input validation
    with pytest.raises(TypeError):
        nurbs_fitting.compute_params_curve("not a list")

@pytest.mark.parametrize('backend', be.list_available_backends())
def test_compute_params_surface(backend, sample_points_surface):
    """Tests the compute_params_surface function."""
    be.set_backend(backend)
    size_u = 4
    size_v = 4

    # Test with default chord length parametrization
    uk, vl = nurbs_fitting.compute_params_surface(sample_points_surface, size_u, size_v)
    assert len(uk) == size_u
    assert len(vl) == size_v
    assert all(isinstance(p, float) for p in uk)
    assert all(isinstance(p, float) for p in vl)
    assert uk[0] == 0.0
    assert uk[-1] == 1.0
    assert vl[0] == 0.0
    assert vl[-1] == 1.0

    # Test with centripetal parametrization
    uk_cent, vl_cent = nurbs_fitting.compute_params_surface(
        sample_points_surface, size_u, size_v, centripetal=True
    )
    assert len(uk_cent) == size_u
    assert len(vl_cent) == size_v

def test_compute_knot_vector():
    """Tests the compute_knot_vector function."""
    degree = 2
    num_dpts = 5
    num_cpts = 5
    params = [0.0, 0.25, 0.5, 0.75, 1.0]

    kv = nurbs_fitting.compute_knot_vector(degree, num_dpts, num_cpts, params)

    # Expected number of knots is num_cpts + degree + 1
    assert len(kv) == num_cpts + degree + 1
    # Check for clamped ends
    assert kv[:degree+1] == [0.0] * (degree + 1)
    assert kv[-degree-1:] == [1.0] * (degree + 1)

def test_build_coeff_matrix():
    """Tests the _build_coeff_matrix private function."""
    degree = 2
    num_points = 4
    knotvector = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
    params = [0.1, 0.4, 0.6, 0.9]
    points = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 0.0]]

    matrix_a = nurbs_fitting._build_coeff_matrix(degree, knotvector, params, points)

    assert isinstance(matrix_a, list)
    assert len(matrix_a) == num_points
    assert all(len(row) == num_points for row in matrix_a)

@pytest.mark.parametrize('backend', be.list_available_backends())
def test_approximate_surface(backend, sample_points_surface):
    """Tests the approximate_surface function."""
    be.set_backend(backend)
    size_u = 4
    size_v = 4
    degree_u = 2
    degree_v = 2

    # Test with default parameters
    result = nurbs_fitting.approximate_surface(
        sample_points_surface, size_u, size_v, degree_u, degree_v
    )
    ctrlpts, d_u, d_v, nc_u, nc_v, kv_u, kv_v = result

    assert d_u == degree_u
    assert d_v == degree_v
    assert isinstance(ctrlpts, list)
    assert len(ctrlpts) == nc_u * nc_v
    assert len(kv_u) == nc_u + d_u + 1
    assert len(kv_v) == nc_v + d_v + 1

    # Test with optional parameters (centripetal and custom ctrlpts size)
    result_optional = nurbs_fitting.approximate_surface(
        sample_points_surface,
        size_u,
        size_v,
        degree_u,
        degree_v,
        centripetal=True,
        ctrlpts_size_u=4,
        ctrlpts_size_v=4,
    )
    _, _, _, nc_u_opt, nc_v_opt, _, _ = result_optional
    assert nc_u_opt == 4
    assert nc_v_opt == 4
