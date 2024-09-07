import pytest
import numpy as np
from optiland.rays import BaseRays, RealRays, ParaxialRays


def test_translate():
    rays = BaseRays()
    rays.x = 1.0
    rays.y = 2.0
    rays.z = 3.0

    rays.translate(0.5, -1.0, 2.5)

    assert rays.x == 1.5
    assert rays.y == 1.0
    assert rays.z == 5.5

    rays.translate(-1.5, 0.0, -5.5)

    assert rays.x == 0.0
    assert rays.y == 1.0
    assert rays.z == 0.0

    rays.translate(0.0, 0.0, 0.0)

    assert rays.x == 0.0
    assert rays.y == 1.0
    assert rays.z == 0.0

    rays.translate(2.0, -1.0, 3.0)

    assert rays.x == 2.0
    assert rays.y == 0.0
    assert rays.z == 3.0


def test__process_input():
    rays = BaseRays()

    # Test scalar input
    data = 1
    processed_data = rays._process_input(data)
    assert isinstance(processed_data, np.ndarray)
    assert processed_data.shape == (1,)
    assert processed_data.dtype == float
    assert processed_data[0] == 1.0

    # Test float input
    data = 2.5
    processed_data = rays._process_input(data)
    assert isinstance(processed_data, np.ndarray)
    assert processed_data.shape == (1,)
    assert processed_data.dtype == float
    assert processed_data[0] == 2.5

    # Test numpy array input
    data = np.array([3, 4, 5])
    processed_data = rays._process_input(data)
    assert isinstance(processed_data, np.ndarray)
    assert processed_data.shape == (3,)
    assert processed_data.dtype == float
    assert np.array_equal(processed_data, np.array([3.0, 4.0, 5.0]))

    # Test unsupported input type
    data = "invalid"
    with pytest.raises(ValueError):
        rays._process_input(data)


def test_real_rays_init():
    x = 1
    y = 2
    z = 3
    L = 4
    M = 5
    N = 6
    intensity = 7
    wavelength = 8

    rays = RealRays(x, y, z, L, M, N, intensity, wavelength)

    assert isinstance(rays.x, np.ndarray)
    assert rays.x.shape == (1,)
    assert rays.x.dtype == float
    assert rays.x[0] == 1.0

    assert isinstance(rays.y, np.ndarray)
    assert rays.y.shape == (1,)
    assert rays.y.dtype == float
    assert rays.y[0] == 2.0

    assert isinstance(rays.z, np.ndarray)
    assert rays.z.shape == (1,)
    assert rays.z.dtype == float
    assert rays.z[0] == 3.0

    assert isinstance(rays.L, np.ndarray)
    assert rays.L.shape == (1,)
    assert rays.L.dtype == float
    assert rays.L[0] == 4.0

    assert isinstance(rays.M, np.ndarray)
    assert rays.M.shape == (1,)
    assert rays.M.dtype == float
    assert rays.M[0] == 5.0

    assert isinstance(rays.N, np.ndarray)
    assert rays.N.shape == (1,)
    assert rays.N.dtype == float
    assert rays.N[0] == 6.0

    assert isinstance(rays.i, np.ndarray)
    assert rays.i.shape == (1,)
    assert rays.i.dtype == float
    assert rays.i[0] == 7.0

    assert isinstance(rays.w, np.ndarray)
    assert rays.w.shape == (1,)
    assert rays.w.dtype == float
    assert rays.w[0] == 8.0

    assert isinstance(rays.opd, np.ndarray)
    assert rays.opd.shape == (1,)
    assert rays.opd.dtype == float
    assert rays.opd[0] == 0.0


def test_rotate_x():
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_x(np.pi / 2)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(-3.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(0.0, abs=1e-10)

    rays.rotate_x(-np.pi / 2)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.rotate_x(np.pi)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(-2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(-3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(-1.0, abs=1e-10)

    rays.rotate_x(0.0)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(-2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(-3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(-1.0, abs=1e-10)


def test_rotate_y():
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_y(np.pi / 2)

    assert rays.x[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(0.0, abs=1e-10)

    rays.rotate_y(-np.pi / 2)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.rotate_y(np.pi)

    assert rays.x[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(-3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(-1.0, abs=1e-10)

    rays.rotate_y(0.0)

    assert rays.x[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(-3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(-1.0, abs=1e-10)


def test_rotate_z():
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_z(np.pi / 2)

    assert rays.x[0] == pytest.approx(-2.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.rotate_z(-np.pi / 2)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.rotate_z(np.pi)

    assert rays.x[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(-2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.rotate_z(0.0)

    assert rays.x[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(-2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)


def test_propagate():
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.propagate(2.0)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(5.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.propagate(-1.5)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.5, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.propagate(0.0)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.5, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.propagate(3.0)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(6.5, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)


def test_clip():
    # Test clipping with condition True
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    condition = True
    rays.clip(condition)
    assert rays.i[0] == pytest.approx(0.0, abs=1e-10)

    # Test clipping with condition False
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    condition = False
    rays.clip(condition)
    assert rays.i[0] == pytest.approx(1.0, abs=1e-10)


def test_paraxial_rays_init():
    y = 1.0
    u = 0.1
    z = -10.0
    wavelength = 2.0

    rays = ParaxialRays(y, u, z, wavelength)

    assert isinstance(rays.x, np.ndarray)
    assert rays.x.shape == (1,)
    assert rays.x.dtype == float
    assert rays.x[0] == pytest.approx(0.0, abs=1e-10)

    assert isinstance(rays.y, np.ndarray)
    assert rays.y.shape == (1,)
    assert rays.y.dtype == float
    assert rays.y[0] == pytest.approx(1.0, abs=1e-10)

    assert isinstance(rays.z, np.ndarray)
    assert rays.z.shape == (1,)
    assert rays.z.dtype == float
    assert rays.z[0] == pytest.approx(-10.0, abs=1e-10)

    assert isinstance(rays.u, np.ndarray)
    assert rays.u.shape == (1,)
    assert rays.u.dtype == float
    assert rays.u[0] == pytest.approx(0.1, abs=1e-10)

    assert isinstance(rays.i, np.ndarray)
    assert rays.i.shape == (1,)
    assert rays.i.dtype == float
    assert rays.i[0] == pytest.approx(1.0, abs=1e-10)

    assert isinstance(rays.w, np.ndarray)
    assert rays.w.shape == (1,)
    assert rays.w.dtype == float
    assert rays.w[0] == pytest.approx(2.0, abs=1e-10)


def test_paraxial_propagate():
    y = 1.0
    u = 0.1
    z = -10.0
    wavelength = 2.0

    rays = ParaxialRays(y, u, z, wavelength)

    rays.propagate(t=0.0)

    assert rays.x[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(-10.0, abs=1e-10)
    assert rays.u[0] == pytest.approx(0.1, abs=1e-10)

    rays.propagate(t=5.0)

    assert rays.x[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(1.5, abs=1e-10)
    assert rays.z[0] == pytest.approx(-5.0, abs=1e-10)
    assert rays.u[0] == pytest.approx(0.1, abs=1e-10)

    rays.propagate(t=-5.0)

    assert rays.x[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(-10.0, abs=1e-10)
    assert rays.u[0] == pytest.approx(0.1, abs=1e-10)


def test_reflect():
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    # Reflect rays on a surface with normal (0, 0, 1)
    rays.reflect(0.0, 0.0, 1.0)

    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(-1.0, abs=1e-10)

    # Reflect rays on a surface with normal (1, 0, 0)
    rays = RealRays(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 1.0)
    rays.reflect(1.0, 0.0, 0.0)

    assert rays.L[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(0.0, abs=1e-10)

    # Reflect rays on a surface with normal (0, 1, 0)
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 1.0)
    rays.reflect(0.0, 1.0, 0.0)

    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(0.0, abs=1e-10)
