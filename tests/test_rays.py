import optiland.backend as be
import pytest

from optiland.rays import (
    BaseRays,
    ParaxialRays,
    PolarizationState,
    PolarizedRays,
    RayGenerator,
    RealRays,
    create_polarization,
)
from optiland.samples.lithography import UVProjectionLens
from optiland.samples.objectives import TessarLens
from tests.utils import assert_allclose


def test_translate(set_test_backend):
    rays = BaseRays()
    rays.x = 1.0
    rays.y = 2.0
    rays.z = 3.0

    rays.translate(0.5, -1.0, 2.5)

    assert_allclose(rays.x, 1.5)
    assert_allclose(rays.y, 1.0)
    assert_allclose(rays.z, 5.5)

    rays.translate(-1.5, 0.0, -5.5)

    assert_allclose(rays.x, 0.0)
    assert_allclose(rays.y, 1.0)
    assert_allclose(rays.z, 0.0)

    rays.translate(0.0, 0.0, 0.0)

    assert_allclose(rays.x, 0.0)
    assert_allclose(rays.y, 1.0)
    assert_allclose(rays.z, 0.0)

    rays.translate(2.0, -1.0, 3.0)

    assert_allclose(rays.x, 2.0)
    assert_allclose(rays.y, 0.0)
    assert_allclose(rays.z, 3.0)


def test__process_input(set_test_backend):
    # Test scalar input
    data = 1
    processed_data = be.as_array_1d(data)
    assert isinstance(processed_data, be.ndarray)
    assert processed_data.shape == (1,)
    assert processed_data.dtype == be.array(data).dtype
    assert_allclose(processed_data[0], 1.0)

    # Test float input
    data = 2.5
    processed_data = be.as_array_1d(data)
    assert isinstance(processed_data, be.ndarray)
    assert processed_data.shape == (1,)
    assert processed_data.dtype == be.array(data).dtype
    assert_allclose(processed_data[0], 2.5)

    # Test numpy array input
    data = be.array([3, 4, 5])
    processed_data = be.as_array_1d(data)
    assert isinstance(processed_data, be.ndarray)
    assert processed_data.shape == (3,)
    assert processed_data.dtype == be.array(data).dtype
    assert be.array_equal(processed_data, be.array([3.0, 4.0, 5.0]))

    # Test unsupported input type
    data = "invalid"
    with pytest.raises(ValueError):
        be.as_array_1d(data)


def test_real_rays_init(set_test_backend):
    x = 1
    y = 2
    z = 3
    L = 4
    M = 5
    N = 6
    intensity = 7
    wavelength = 8

    rays = RealRays(x, y, z, L, M, N, intensity, wavelength)

    assert isinstance(rays.x, be.ndarray)
    assert rays.x.shape == (1,)
    assert rays.x.dtype == be.array(x).dtype
    assert_allclose(rays.x[0], 1.0)

    assert isinstance(rays.y, be.ndarray)
    assert rays.y.shape == (1,)
    assert rays.y.dtype == be.array(y).dtype
    assert_allclose(rays.y[0], 2.0)

    assert isinstance(rays.z, be.ndarray)
    assert rays.z.shape == (1,)
    assert rays.z.dtype == be.array(z).dtype
    assert_allclose(rays.z[0], 3.0)

    assert isinstance(rays.L, be.ndarray)
    assert rays.L.shape == (1,)
    assert rays.L.dtype == be.array(L).dtype
    assert_allclose(rays.L[0], 4.0)

    assert isinstance(rays.M, be.ndarray)
    assert rays.M.shape == (1,)
    assert rays.M.dtype == be.array(M).dtype
    assert_allclose(rays.M[0], 5.0)

    assert isinstance(rays.N, be.ndarray)
    assert rays.N.shape == (1,)
    assert rays.N.dtype == be.array(N).dtype
    assert_allclose(rays.N[0], 6.0)

    assert isinstance(rays.i, be.ndarray)
    assert rays.i.shape == (1,)
    assert rays.i.dtype == be.array(intensity).dtype
    assert_allclose(rays.i[0], 7.0)

    assert isinstance(rays.w, be.ndarray)
    assert rays.w.shape == (1,)
    assert rays.w.dtype == be.array(wavelength).dtype
    assert_allclose(rays.w[0], 8.0)

    assert isinstance(rays.opd, be.ndarray)
    assert rays.opd.shape == (1,)
    assert rays.opd.dtype == be.array(0.0).dtype
    assert_allclose(rays.opd[0], 0.0)


def test_rotate_x(set_test_backend):
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_x(be.pi / 2)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 0.0, rtol=0, atol=1e-10)

    rays.rotate_x(-be.pi / 2)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_x(be.pi)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, rtol=0, atol=1e-10)

    rays.rotate_x(0.0)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, rtol=0, atol=1e-10)


def test_rotate_y(set_test_backend):
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_y(be.pi / 2)

    assert_allclose(rays.x[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 0.0, rtol=0, atol=1e-10)

    rays.rotate_y(-be.pi / 2)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_y(be.pi)

    assert_allclose(rays.x[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, rtol=0, atol=1e-10)

    rays.rotate_y(0.0)

    assert_allclose(rays.x[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], -3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, rtol=0, atol=1e-10)


def test_rotate_z(set_test_backend):
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_z(be.pi / 2)

    assert_allclose(rays.x[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_z(-be.pi / 2)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_z(be.pi)

    assert_allclose(rays.x[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.rotate_z(0.0)

    assert_allclose(rays.x[0], -1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], -2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)


def test_propagate(set_test_backend):
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.propagate(2.0)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 5.0, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.propagate(-1.5)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.5, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.propagate(0.0)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 3.5, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)

    rays.propagate(3.0)

    assert_allclose(rays.x[0], 1.0, rtol=0, atol=1e-10)
    assert_allclose(rays.y[0], 2.0, rtol=0, atol=1e-10)
    assert_allclose(rays.z[0], 6.5, rtol=0, atol=1e-10)
    assert_allclose(rays.L[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, rtol=0, atol=1e-10)
    assert_allclose(rays.N[0], 1.0, rtol=0, atol=1e-10)


def test_clip(set_test_backend):
    # Test clipping with condition True
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    condition = True
    rays.clip(condition)
    assert_allclose(rays.i[0], 0.0, atol=1e-10)

    # Test clipping with condition False
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    condition = False
    rays.clip(condition)
    assert_allclose(rays.i[0], 1.0, atol=1e-10)


@pytest.mark.usefixtures("set_test_backend")
def test_refract():
    """Tests the RealRays.refract() method."""
    # Case 1: Normal Incidence
    rays_normal = RealRays(x=0, y=0, z=0, L=0, M=0, N=1, intensity=1, wavelength=0.5)
    nx, ny, nz = 0.0, 0.0, 1.0
    n1, n2 = 1.0, 1.5
    rays_normal.refract(be.array(nx), be.array(ny), be.array(nz), n1, n2)
    assert_allclose(rays_normal.L, be.array([0.0]), atol=1e-10)
    assert_allclose(rays_normal.M, be.array([0.0]), atol=1e-10)
    assert_allclose(rays_normal.N, be.array([1.0]), atol=1e-10)

    # Case 2: Angled Incidence (No TIR)
    # Angle of incidence = 30 deg.
    theta1_rad = be.radians(be.array(30.0))
    rays_angled = RealRays(
        x=0,
        y=0,
        z=0,
        L=0,
        M=be.sin(theta1_rad),
        N=be.cos(theta1_rad),
        intensity=1,
        wavelength=0.5,
    )
    n1_angled, n2_angled = 1.0, 1.5
    rays_angled.refract(be.array(nx), be.array(ny), be.array(nz), n1_angled, n2_angled)

    # Expected refracted angle using Snell's Law: sin(theta2) = (n1/n2) * sin(theta1)
    sin_theta2 = (n1_angled / n2_angled) * be.sin(theta1_rad)
    theta2_rad = be.arcsin(sin_theta2)
    M_expected = be.sin(theta2_rad)
    N_expected = be.cos(theta2_rad)
    assert_allclose(rays_angled.M, M_expected, atol=1e-10)
    assert_allclose(rays_angled.N, N_expected, atol=1e-10)
    assert_allclose(rays_angled.L, be.array([0.0]), atol=1e-10) # L should remain 0

    # Case 3: Total Internal Reflection (TIR)
    # Angle of incidence = 60 deg. n1=1.5 (glass), n2=1.0 (air).
    # Critical angle for 1.5->1.0 is asin(1/1.5) approx 41.8 deg. So 60 deg incidence will TIR.
    theta1_tir_rad = be.radians(be.array(60.0))
    rays_tir = RealRays(
        x=0,
        y=0,
        z=0,
        L=0,
        M=be.sin(theta1_tir_rad),
        N=be.cos(theta1_tir_rad),
        intensity=1,
        wavelength=0.5,
    )
    n1_tir, n2_tir = 1.5, 1.0
    rays_tir.refract(be.array(nx), be.array(ny), be.array(nz), n1_tir, n2_tir)
    # Expect NaN for direction cosines due to TIR
    assert be.all(be.isnan(rays_tir.L))
    assert be.all(be.isnan(rays_tir.M))
    assert be.all(be.isnan(rays_tir.N))


def test_paraxial_rays_init(set_test_backend):
    y = 1.0
    u = 0.1
    z = -10.0
    wavelength = 2.0

    rays = ParaxialRays(y, u, z, wavelength)

    assert isinstance(rays.x, be.ndarray)
    assert rays.x.shape == (1,)
    assert rays.x.dtype == be.array(0.0).dtype
    assert_allclose(rays.x[0], 0.0, atol=1e-10)

    assert isinstance(rays.y, be.ndarray)
    assert rays.y.shape == (1,)
    assert rays.y.dtype == be.array(y).dtype
    assert_allclose(rays.y[0], 1.0, atol=1e-10)

    assert isinstance(rays.z, be.ndarray)
    assert rays.z.shape == (1,)
    assert rays.z.dtype == be.array(z).dtype
    assert_allclose(rays.z[0], -10.0, atol=1e-10)

    assert isinstance(rays.u, be.ndarray)
    assert rays.u.shape == (1,)
    assert rays.u.dtype == be.array(u).dtype
    assert_allclose(rays.u[0], 0.1, atol=1e-10)

    assert isinstance(rays.i, be.ndarray)
    assert rays.i.shape == (1,)
    assert rays.i.dtype == be.array(1.0).dtype
    assert_allclose(rays.i[0], 1.0, atol=1e-10)

    assert isinstance(rays.w, be.ndarray)
    assert rays.w.shape == (1,)
    assert rays.w.dtype == be.array(wavelength).dtype
    assert_allclose(rays.w[0], 2.0, atol=1e-10)


def test_paraxial_propagate(set_test_backend):
    y = 1.0
    u = 0.1
    z = -10.0
    wavelength = 2.0

    rays = ParaxialRays(y, u, z, wavelength)

    rays.propagate(t=0.0)

    assert_allclose(rays.x[0], 0.0, atol=1e-10)
    assert_allclose(rays.y[0], 1.0, atol=1e-10)
    assert_allclose(rays.z[0], -10.0, atol=1e-10)
    assert_allclose(rays.u[0], 0.1, atol=1e-10)

    rays.propagate(t=5.0)

    assert_allclose(rays.x[0], 0.0, atol=1e-10)
    assert_allclose(rays.y[0], 1.5, atol=1e-10)
    assert_allclose(rays.z[0], -5.0, atol=1e-10)
    assert_allclose(rays.u[0], 0.1, atol=1e-10)

    rays.propagate(t=-5.0)

    assert_allclose(rays.x[0], 0.0, atol=1e-10)
    assert_allclose(rays.y[0], 1.0, atol=1e-10)
    assert_allclose(rays.z[0], -10.0, atol=1e-10)
    assert_allclose(rays.u[0], 0.1, atol=1e-10)


@pytest.mark.usefixtures("set_test_backend")
def test_paraxial_rays_rotate():
    """Tests the rotation methods of the ParaxialRays class."""
    # Test rotate_x()
    y_init, u_init, z_init = 1.0, 0.1, 10.0
    rays = ParaxialRays(y=y_init, u=u_init, z=z_init, wavelength=0.5)

    # Rotate by pi/4
    rx_angle = be.pi / 4.0
    rays.rotate_x(rx_angle) # Use a different variable name to avoid conflict with rays.x

    y_expected = y_init * be.cos(rx_angle) - z_init * be.sin(rx_angle)
    z_expected = y_init * be.sin(rx_angle) + z_init * be.cos(rx_angle)
    M_expected = u_init * be.cos(rx_angle) - be.sin(rx_angle)
    N_expected = u_init * be.sin(rx_angle) + be.cos(rx_angle)
    u_expected = M_expected / N_expected

    assert_allclose(rays.y[0], y_expected, atol=1e-10)
    assert_allclose(rays.z[0], z_expected, atol=1e-10)
    assert_allclose(rays.u[0], u_expected, atol=1e-10)

    # Test rotate_x(0)
    rays_zero_rot = ParaxialRays(y=y_init, u=u_init, z=z_init, wavelength=0.5) # New instance
    rays_zero_rot.rotate_x(0.0)
    assert_allclose(rays_zero_rot.y[0], y_init, atol=1e-10)
    assert_allclose(rays_zero_rot.z[0], z_init, atol=1e-10)
    assert_allclose(rays_zero_rot.u[0], u_init, atol=1e-10)

    # Test rotate_y()
    rays_y_rot = ParaxialRays(y=1.0, u=0.1, z=10.0, wavelength=0.5) # New instance
    with pytest.raises(NotImplementedError, match="rotate_y is not applicable"):
        rays_y_rot.rotate_y(0.1)

    # Test rotate_z()
    rays_z_rot = ParaxialRays(y=1.0, u=0.1, z=10.0, wavelength=0.5) # New instance
    with pytest.raises(NotImplementedError, match="rotate_z is not applicable"):
        rays_z_rot.rotate_z(0.1)


def test_reflect(set_test_backend):
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    # Reflect rays on a surface with normal (0, 0, 1)
    rays.reflect(0.0, 0.0, 1.0)

    assert_allclose(rays.L[0], 0.0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, atol=1e-10)
    assert_allclose(rays.N[0], -1.0, atol=1e-10)

    # Reflect rays on a surface with normal (1, 0, 0)
    rays = RealRays(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 1.0)
    rays.reflect(1.0, 0.0, 0.0)

    assert_allclose(rays.L[0], -1.0, atol=1e-10)
    assert_allclose(rays.M[0], 0.0, atol=1e-10)
    assert_allclose(rays.N[0], 0.0, atol=1e-10)

    # Reflect rays on a surface with normal (0, 1, 0)
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 1.0)
    rays.reflect(0.0, 1.0, 0.0)

    assert_allclose(rays.L[0], 0.0, atol=1e-10)
    assert_allclose(rays.M[0], -1.0, atol=1e-10)
    assert_allclose(rays.N[0], 0.0, atol=1e-10)


def test_real_rays_str(set_test_backend):
    """Tests the __str__ method of the RealRays class."""

    rays_empty = RealRays([], [], [], [], [], [], [], [])
    rays_empty.x = [] # Ensure x is explicitly empty for the check
    assert str(rays_empty) == "RealRays object (No rays)"

    x = be.array([1.0, 1.1])
    y = be.array([2.0, 2.1])
    z = be.array([3.0, 3.1])
    L = be.array([0.0, 0.1])
    M = be.array([0.0, 0.2])
    N = be.array([1.0, 0.9])
    i = be.array([1.0, 0.5])
    w = be.array([0.55, 0.65])
    rays_few = RealRays(x, y, z, L, M, N, i, w)

    h = " Ray # |          x |          y |          z |          L |          M |          N |  Intensity |   Wavelength\n"
    s = "----------------------------------------------------------------------------------------------------------------------\n"
    r0_text = "     0 |     1.0000 |     2.0000 |     3.0000 |   0.000000 |   0.000000 |   1.000000 |     1.0000 |       0.5500\n"
    r1_text = "     1 |     1.1000 |     2.1000 |     3.1000 |   0.100000 |   0.200000 |   0.900000 |     0.5000 |       0.6500\n"
    f_text = "Showing 2 of 2 rays.\n"
    expected_few = h + s + r0_text + r1_text + s + f_text
    assert str(rays_few) == expected_few

    x_many = be.array([1.0, 1.1, 1.2, 1.3, 1.4])
    y_many = be.array([2.0, 2.1, 2.2, 2.3, 2.4])
    z_many = be.array([3.0, 3.1, 3.2, 3.3, 3.4])
    L_many = be.array([0.0, 0.1, 0.2, 0.3, 0.4])
    M_many = be.array([0.0, 0.1, 0.2, 0.3, 0.4])
    N_many = be.array([1.0, 0.9, 0.8, 0.7, 0.6])
    i_many = be.array([1.0, 0.9, 0.8, 0.7, 0.6])
    w_many = be.array([0.55, 0.55, 0.55, 0.55, 0.55])
    rays_many = RealRays(x_many, y_many, z_many, L_many, M_many, N_many, i_many, w_many)

    r0_many = "     0 |     1.0000 |     2.0000 |     3.0000 |   0.000000 |   0.000000 |   1.000000 |     1.0000 |       0.5500\n"
    r2_many = "     2 |     1.2000 |     2.2000 |     3.2000 |   0.200000 |   0.200000 |   0.800000 |     0.8000 |       0.5500\n"
    r4_many = "     4 |     1.4000 |     2.4000 |     3.4000 |   0.400000 |   0.400000 |   0.600000 |     0.6000 |       0.5500\n"
    f_many = "Showing 3 of 5 rays.\n"
    expected_many = h + s + r0_many + r2_many + r4_many + s + f_many
    assert str(rays_many) == expected_many


class TestPolarizationState:
    def test_constructor(self, set_test_backend):
        state = PolarizationState(is_polarized=True, Ex=1, Ey=2, phase_x=0, phase_y=1)
        assert state.is_polarized is True
        assert_allclose(state.Ex, 1 / be.sqrt(5))
        assert_allclose(state.Ey, 2 / be.sqrt(5))
        assert state.phase_x == 0
        assert state.phase_y == 1

    def test_constructor_invalid(self, set_test_backend):
        with pytest.raises(ValueError):
            PolarizationState(is_polarized=True, Ex=1, Ey=2, phase_x=0, phase_y=None)
        with pytest.raises(ValueError):
            PolarizationState(is_polarized=False, Ex=1, Ey=2, phase_x=None, phase_y=1)

    def test_str(self, set_test_backend):
        state = PolarizationState(is_polarized=True, Ex=1, Ey=0, phase_x=0, phase_y=1)
        # Format numbers to match the __str__ output which uses default float formatting
        ex_str = f"{be.to_numpy(state.Ex)[0]:.1f}" if be.is_tensor(state.Ex) else f"{state.Ex:.1f}"
        ey_str = f"{be.to_numpy(state.Ey)[0]:.1f}" if be.is_tensor(state.Ey) else f"{state.Ey:.1f}"
        px_str = f"{be.to_numpy(state.phase_x)[0]:.1f}" if be.is_tensor(state.phase_x) else f"{state.phase_x:.1f}"
        py_str = f"{be.to_numpy(state.phase_y)[0]:.1f}" if be.is_tensor(state.phase_y) else f"{state.phase_y:.1f}"
        val = f"Polarized Light: Ex: {ex_str}, Ey: {ey_str}, Phase x: {px_str}, Phase y: {py_str}"
        assert str(state) == val


    def test_str_unpolarized(self, set_test_backend):
        state = PolarizationState(is_polarized=False)
        assert str(state) == "Unpolarized Light"

    def test_repr(self, set_test_backend):
        state = PolarizationState(is_polarized=True, Ex=0, Ey=1, phase_x=0, phase_y=1)
        ex_str = f"{be.to_numpy(state.Ex)[0]:.1f}" if be.is_tensor(state.Ex) else f"{state.Ex:.1f}"
        ey_str = f"{be.to_numpy(state.Ey)[0]:.1f}" if be.is_tensor(state.Ey) else f"{state.Ey:.1f}"
        px_str = f"{be.to_numpy(state.phase_x)[0]:.1f}" if be.is_tensor(state.phase_x) else f"{state.phase_x:.1f}"
        py_str = f"{be.to_numpy(state.phase_y)[0]:.1f}" if be.is_tensor(state.phase_y) else f"{state.phase_y:.1f}"
        val = f"Polarized Light: Ex: {ex_str}, Ey: {ey_str}, Phase x: {px_str}, Phase y: {py_str}"
        assert repr(state) == val


class TestCreatePolarization:
    def test_create_polarization_unpolarized(self, set_test_backend):
        state = create_polarization("unpolarized")
        assert state.is_polarized is False
        assert state.Ex is None
        assert state.Ey is None
        assert state.phase_x is None
        assert state.phase_y is None

    def test_create_polarization_horizontal(self, set_test_backend):
        state = create_polarization("H")
        assert state.is_polarized is True
        assert_allclose(state.Ex, 1.0, atol=1e-10)
        assert_allclose(state.Ey, 0.0, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, 0.0, atol=1e-10)

    def test_create_polarization_vertical(self, set_test_backend):
        state = create_polarization("V")
        assert state.is_polarized is True
        assert_allclose(state.Ex, 0.0, atol=1e-10)
        assert_allclose(state.Ey, 1.0, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, 0.0, atol=1e-10)

    def test_create_polarization_linear_45(self, set_test_backend):
        state = create_polarization("L+45")
        assert state.is_polarized is True
        assert_allclose(state.Ex, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.Ey, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, 0.0, atol=1e-10)

    def test_create_polarization_linear_minus_45(self, set_test_backend):
        state = create_polarization("L-45")
        assert state.is_polarized is True
        assert_allclose(state.Ex, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.Ey, -be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, 0.0, atol=1e-10)

    def test_create_polarization_right_circular(self, set_test_backend):
        state = create_polarization("RCP")
        assert state.is_polarized is True
        assert_allclose(state.Ex, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.Ey, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, -be.pi / 2, atol=1e-10)

    def test_create_polarization_left_circular(self, set_test_backend):
        state = create_polarization("LCP")
        assert state.is_polarized is True
        assert_allclose(state.Ex, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.Ey, be.sqrt(2) / 2, atol=1e-10)
        assert_allclose(state.phase_x, 0.0, atol=1e-10)
        assert_allclose(state.phase_y, be.pi / 2, atol=1e-10)

    def test_create_polarization_invalid(self, set_test_backend):
        with pytest.raises(ValueError):
            create_polarization("invalid")


class TestPolarizedRays:
    def test_init(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)

        assert isinstance(rays.p, be.ndarray)
        assert rays.p.shape == (1, 3, 3)
        assert_allclose(rays.p[0], be.eye(3))

    def test_get_output_field(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
        E = be.array([[1.0, 0.0, 0.0]])

        output_field = rays.get_output_field(E)
        assert output_field.shape == (1, 3)
        assert_allclose(output_field, E)

    def test_update_intensity(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L_init = be.array([0.0])
        M_init = be.array([0.0])
        N_init = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L_init, M_init, N_init, intensity, wavelength)
        # _get_3d_electric_field uses _L0, _M0, _N0 so they must be initialized
        rays._L0 = be.copy(L_init)
        rays._M0 = be.copy(M_init)
        rays._N0 = be.copy(N_init)

        state = PolarizationState(
            is_polarized=True,
            Ex=1.0,
            Ey=0.0,
            phase_x=0.0,
            phase_y=0.0,
        )

        rays.update_intensity(state)
        assert rays.i.shape == (1,)
        assert_allclose(rays.i[0], 1.0, atol=1e-10)

        # test case for unpolarized light
        state_unpol = PolarizationState(is_polarized=False)
        rays.update_intensity(state_unpol) # Intensity should remain 1.0 as p is identity
        assert rays.i.shape == (1,)
        assert_allclose(rays.i[0], 1.0, atol=1e-10)


    def test_update(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
        rays.L0 = be.array([0.0]) # L0, M0, N0 are previous direction
        rays.M0 = be.array([0.0])
        rays.N0 = be.array([1.0])
        # L, M, N are current direction (after surface interaction, assume no change for this test)
        rays.L = be.array([0.0])
        rays.M = be.array([0.0])
        rays.N = be.array([1.0])

        jones_matrix_identity = be.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=be.complex64)
        # For k0 parallel to k1, s=be.cross(k0,k1) is zero.
        # The fallback for s when mag=0 is cross(k0, [1,0,0]). If k0 is [0,0,1], s becomes [0,-1,0]
        # p0 = k0 x s = [0,0,1] x [0,-1,0] = [1,0,0]
        # p1 = k1 x s = [0,0,1] x [0,-1,0] = [1,0,0]
        # o_in = stack(s,p0,k0) = [[0,-1,0],[1,0,0],[0,0,1]]
        # o_out = stack(s,p1,k1).T = [[0,1,0],[-1,0,0],[0,0,1]] (transpose for stack axis=2)
        # If k0=k1=[0,0,1] and jones_matrix is identity, p should be identity
        rays.update(jones_matrix_identity)
        assert rays.p.shape == (1, 3, 3)
        assert_allclose(rays.p, jones_matrix_identity, atol=1e-7)

        # test case when k not orthogonal to N0 (L0, M0, N0 are k_incident, L, M, N are k_transmitted)
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.1])
        rays.N0 = be.sqrt(1.0 - 0.1**2)
        rays.L = be.array([0.0]) # K_transmitted, assume it's same for simplicity here
        rays.M = be.array([0.1])
        rays.N = be.sqrt(1.0 - 0.1**2)

        rays.p = be.tile(be.eye(3), (be.size(rays.x), 1, 1)) # Reset polarization matrix
        rays.update(jones_matrix_identity) # Should still result in identity because k0 nearly parallel to k1
        assert rays.p.shape == (1, 3, 3)
        assert_allclose(rays.p, jones_matrix_identity, atol=1e-7)


        # test case when jones = None (should also result in identity if k0=k1)
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.0])
        rays.N0 = be.array([1.0])
        rays.L = be.array([0.0])
        rays.M = be.array([0.0])
        rays.N = be.array([1.0])
        rays.p = be.tile(be.eye(3), (be.size(rays.x), 1, 1)) # Reset
        rays.update(None)
        assert rays.p.shape == (1, 3, 3)
        assert_allclose(rays.p, be.eye(3), atol=1e-7)


    def test_get_3d_electric_field(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L_init = be.array([0.0]) # k along N
        M_init = be.array([0.0])
        N_init = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L_init, M_init, N_init, intensity, wavelength)
        # Initialize _L0, _M0, _N0 as they are used by _get_3d_electric_field
        rays._L0 = be.copy(L_init)
        rays._M0 = be.copy(M_init)
        rays._N0 = be.copy(N_init)

        state = PolarizationState(
            is_polarized=True,
            Ex=1.0, # s-component
            Ey=0.0, # p-component
            phase_x=0.0,
            phase_y=0.0,
        )

        E = rays._get_3d_electric_field(state)
        assert E.shape == (1, 3)
        # For k along N=[0,0,1], x_direction=[1,0,0]. p_initial = k x x_direction = [0,1,0] (y-axis)
        # p_initial is normalized to p_for_non_parallel = [0,1,0]
        # s_for_non_parallel = p_for_non_parallel x k = [0,1,0] x [0,0,1] = [1,0,0] (x-axis)
        # E = Ex * s + Ey * p = 1.0 * [1,0,0] + 0.0 * [0,1,0] = [1,0,0]
        assert_allclose(E, be.array([[1.0, 0.0, 0.0]]), atol=1e-10)

    def test_get_3d_electric_field_k_parallel_x(self, set_test_backend):
        """Tests _get_3d_electric_field when k is parallel to the x-axis."""
        x_coord, y_coord, z_coord = be.array([0.0]), be.array([0.0]), be.array([0.0])
        intensity, wavelength = be.array([1.0]), be.array([0.5])

        # Case 1: k along +x
        L_plus_x, M_plus_x, N_plus_x = be.array([1.0]), be.array([0.0]), be.array([0.0])
        rays_plus_x = PolarizedRays(
            x_coord, y_coord, z_coord, L_plus_x, M_plus_x, N_plus_x, intensity, wavelength
        )
        rays_plus_x._L0, rays_plus_x._M0, rays_plus_x._N0 = L_plus_x, M_plus_x, N_plus_x

        state_Ex1_Ey0 = PolarizationState(is_polarized=True, Ex=1.0, Ey=0.0, phase_x=0.0, phase_y=0.0)
        E_plus_x_Ex1 = rays_plus_x._get_3d_electric_field(state_Ex1_Ey0)
        # k=[1,0,0] (parallel to x).
        # s_for_parallel is y_direction [0,1,0].
        # p_for_parallel is k x s_for_parallel = [1,0,0] x [0,1,0] = [0,0,1].
        # E = Ex * s + Ey * p = 1.0 * [0,1,0] + 0.0 * [0,0,1] = [0,1,0]
        assert_allclose(E_plus_x_Ex1, be.array([[0.0, 1.0, 0.0]]), atol=1e-10)

        state_Ex0_Ey1 = PolarizationState(is_polarized=True, Ex=0.0, Ey=1.0, phase_x=0.0, phase_y=0.0)
        E_plus_x_Ey1 = rays_plus_x._get_3d_electric_field(state_Ex0_Ey1)
        # E = Ex * s + Ey * p = 0.0 * [0,1,0] + 1.0 * [0,0,1] = [0,0,1]
        assert_allclose(E_plus_x_Ey1, be.array([[0.0, 0.0, 1.0]]), atol=1e-10)

        # Case 2: k along -x
        L_minus_x, M_minus_x, N_minus_x = be.array([-1.0]), be.array([0.0]), be.array([0.0])
        rays_minus_x = PolarizedRays(
            x_coord, y_coord, z_coord, L_minus_x, M_minus_x, N_minus_x, intensity, wavelength
        )
        rays_minus_x._L0, rays_minus_x._M0, rays_minus_x._N0 = L_minus_x, M_minus_x, N_minus_x

        E_minus_x_Ex1 = rays_minus_x._get_3d_electric_field(state_Ex1_Ey0)
        # k=[-1,0,0] (parallel to x).
        # s_for_parallel is y_direction [0,1,0].
        # p_for_parallel is k x s_for_parallel = [-1,0,0] x [0,1,0] = [0,0,-1].
        # E = Ex * s + Ey * p = 1.0 * [0,1,0] + 0.0 * [0,0,-1] = [0,1,0]
        assert_allclose(E_minus_x_Ex1, be.array([[0.0, 1.0, 0.0]]), atol=1e-10)

        E_minus_x_Ey1 = rays_minus_x._get_3d_electric_field(state_Ex0_Ey1)
        # E = Ex * s + Ey * p = 0.0 * [0,1,0] + 1.0 * [0,0,-1] = [0,0,-1]
        assert_allclose(E_minus_x_Ey1, be.array([[0.0, 0.0, -1.0]]), atol=1e-10)


@pytest.mark.usefixtures("set_test_backend")
class TestRayGenerator:
    def test_generate_rays(self):
        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        lens = TessarLens()
        generator = RayGenerator(lens)
        rays = generator.generate_rays(Hx, Hy, Px, Py, wavelength)

        assert isinstance(rays, RealRays)
        assert rays.x.shape == (2,)
        assert rays.y.shape == (2,)
        assert rays.z.shape == (2,)
        assert rays.L.shape == (2,)
        assert rays.M.shape == (2,)
        assert rays.N.shape == (2,)
        assert rays.i.shape == (2,)
        assert rays.w.shape == (2,)

        assert_allclose(rays.x, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert_allclose(rays.y, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert_allclose(rays.z, be.array([-0.88839505, -0.88839505]), atol=1e-8)
        assert_allclose(rays.L, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert_allclose(rays.M, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert_allclose(rays.N, be.array([0.96882189, 0.96882189]), atol=1e-8)
        assert_allclose(rays.i, be.array([1.0, 1.0]), atol=1e-8)
        assert_allclose(rays.w, be.array([0.55, 0.55]), atol=1e-8)

    def test_generate_rays_telecentric(self):
        lens = UVProjectionLens()
        generator = RayGenerator(lens)

        Hx = 0.0
        Hy = 1.0
        Px = 0.8
        Py = 0.0
        wavelength = 0.248

        rays = generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        assert_allclose(rays.x[0], 0.0, atol=1e-8)
        assert_allclose(rays.y[0], 48.0, atol=1e-8)
        assert_allclose(rays.z[0], -110.85883544, atol=1e-8)
        assert_allclose(rays.L[0], 0.10674041, atol=1e-8)
        assert_allclose(rays.M[0], 0.0, atol=1e-8)
        assert_allclose(rays.N[0], 0.99428692, atol=1e-8)
        assert_allclose(rays.i[0], 1.0, atol=1e-8)
        assert_allclose(rays.w[0], 0.248, atol=1e-8)

    def test_generate_rays_invalid_field_type(self):
        lens = UVProjectionLens()
        generator = RayGenerator(lens)

        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        lens.set_aperture("EPD", 1.0)
        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)

        lens.set_aperture("imageFNO", 10.0)
        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)

        lens.set_field_type("angle")
        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)

    def test_invalid_polarization(self):
        lens = TessarLens()
        lens.surface_group.set_fresnel_coatings()
        generator = RayGenerator(lens)

        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        # polarization type is 'ignore' but lens has coatings
        with pytest.raises(ValueError):
            generator.generate_rays(Hx, Hy, Px, Py, wavelength)

    def test_generate_polarized_rays(self):
        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        wavelength = 0.55

        lens = TessarLens()
        state = PolarizationState(is_polarized=False)
        lens.set_polarization(state)
        generator = RayGenerator(lens)
        rays = generator.generate_rays(Hx, Hy, Px, Py, wavelength)

        assert isinstance(rays, PolarizedRays)
        assert rays.x.shape == (2,)
        assert rays.y.shape == (2,)
        assert rays.z.shape == (2,)
        assert rays.L.shape == (2,)
        assert rays.M.shape == (2,)
        assert rays.N.shape == (2,)
        assert rays.i.shape == (2,)
        assert rays.w.shape == (2,)
        assert rays.p.shape == (2, 3, 3)

        assert_allclose(rays.x, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert_allclose(rays.y, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert_allclose(rays.z, be.array([-0.88839505, -0.88839505]), atol=1e-8)
        assert_allclose(rays.L, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert_allclose(rays.M, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert_allclose(rays.N, be.array([0.96882189, 0.96882189]), atol=1e-8)
        assert_allclose(rays.i, be.array([1.0, 1.0]), atol=1e-8)
        assert_allclose(rays.w, be.array([0.55, 0.55]), atol=1e-8)

    def test_get_ray_origins_infinite_object(self):
        lens = TessarLens()
        generator = RayGenerator(lens)

        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        vx = 0.0
        vy = 0.0

        x0, y0, z0 = generator._get_ray_origins(Hx, Hy, Px, Py, vx, vy)

        assert x0.shape == (2,)
        assert y0.shape == (2,)
        assert z0.shape == (2,)

    def test_get_ray_origins_invalid_field_type(self):
        lens = TessarLens()
        lens.set_field_type("object_height")
        generator = RayGenerator(lens)

        Hx = 0.5
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        vx = 0.0
        vy = 0.1

        with pytest.raises(ValueError):
            generator._get_ray_origins(Hx, Hy, Px, Py, vx, vy)

    def test_invalid_ray_origin_telecentric(self):
        lens = TessarLens()
        lens.obj_space_telecentric = True
        generator = RayGenerator(lens)

        Hx = 0.0
        Hy = 0.5
        Px = be.array([0.1, 0.2])
        Py = be.array([0.1, 0.2])
        vx = 0.0
        vy = 0.1

        with pytest.raises(ValueError):
            generator._get_ray_origins(Hx, Hy, Px, Py, vx, vy)

    def test_normalize(self):
        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        # normalize during propagation
        rays.is_normalized = False
        rays.propagate(1.0)
        assert rays.is_normalized is True

        # manually normalize
        rays.is_normalized = False
        rays.normalize()
        assert rays.is_normalized is True
