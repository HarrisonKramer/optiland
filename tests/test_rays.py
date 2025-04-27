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
        val = "Polarized Light: Ex: 1.0, Ey: 0.0, Phase x: 0.0, Phase y: 1.0"
        assert str(state) == val

    def test_str_unpolarized(self, set_test_backend):
        state = PolarizationState(is_polarized=False)
        assert str(state) == "Unpolarized Light"

    def test_repr(self, set_test_backend):
        state = PolarizationState(is_polarized=True, Ex=0, Ey=1, phase_x=0, phase_y=1)
        val = "Polarized Light: Ex: 0.0, Ey: 1.0, Phase x: 0.0, Phase y: 1.0"
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
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
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
        state = PolarizationState(is_polarized=False)
        rays.update_intensity(state)
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
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.0])
        rays.N0 = be.array([1.0])
        jones_matrix = be.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        rays.update(jones_matrix)
        assert rays.p.shape == (1, 3, 3)
        assert_allclose(rays.p, jones_matrix)

        # test case when k not orthogonal to N0
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.1])
        rays.N0 = be.sqrt(be.array([1 - 0.1**2]))
        rays.update(jones_matrix)
        assert rays.p.shape == (1, 3, 3)
        expected_jones_matrix = be.array(
            [[[1.0, 0.0, 0.0], [0.0, 0.99498744, -0.1], [0.0, 0.1, 0.99498744]]],
        )
        assert_allclose(rays.p, expected_jones_matrix, atol=1e-8) # Reduced tolerance slightly for potential backend differences

        # test case when jones = None
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.0])
        rays.N0 = be.array([1.0])
        rays.update(None)
        assert rays.p.shape == (1, 3, 3)
        # The state should not change if jones_matrix is None, so compare to previous state
        assert_allclose(rays.p, expected_jones_matrix, atol=1e-8)

    def test_get_3d_electric_field(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
        state = PolarizationState(
            is_polarized=True,
            Ex=1.0,
            Ey=0.0,
            phase_x=0.0,
            phase_y=0.0,
        )

        E = rays._get_3d_electric_field(state)
        assert E.shape == (1, 3)
        assert_allclose(E, be.array([[1.0, 0.0, 0.0]]), atol=1e-10)

    def test_get_3d_electric_field_error(self, set_test_backend):
        x = be.array([1.0])
        y = be.array([2.0])
        z = be.array([3.0])
        L = be.array([1.0])  # k-vector propagates in x-direction
        M = be.array([0.0])
        N = be.array([0.0])
        intensity = be.array([1.0])
        wavelength = be.array([1.0])

        rays = PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
        rays._L0 = be.array([1.0])
        rays._M0 = be.array([0.0])
        rays._N0 = be.array([0.0])
        state = PolarizationState(is_polarized=True, Ex=1, Ey=0, phase_x=0, phase_y=0)

        with pytest.raises(ValueError):
            rays._get_3d_electric_field(state)

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
