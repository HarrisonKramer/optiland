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
    assert isinstance(processed_data, be.ndarray)
    assert processed_data.shape == (1,)
    assert processed_data.dtype == float
    assert processed_data[0] == 1.0

    # Test float input
    data = 2.5
    processed_data = rays._process_input(data)
    assert isinstance(processed_data, be.ndarray)
    assert processed_data.shape == (1,)
    assert processed_data.dtype == float
    assert processed_data[0] == 2.5

    # Test numpy array input
    data = be.array([3, 4, 5])
    processed_data = rays._process_input(data)
    assert isinstance(processed_data, be.ndarray)
    assert processed_data.shape == (3,)
    assert processed_data.dtype == float
    assert be.array_equal(processed_data, be.array([3.0, 4.0, 5.0]))

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

    assert isinstance(rays.x, be.ndarray)
    assert rays.x.shape == (1,)
    assert rays.x.dtype == float
    assert rays.x[0] == 1.0

    assert isinstance(rays.y, be.ndarray)
    assert rays.y.shape == (1,)
    assert rays.y.dtype == float
    assert rays.y[0] == 2.0

    assert isinstance(rays.z, be.ndarray)
    assert rays.z.shape == (1,)
    assert rays.z.dtype == float
    assert rays.z[0] == 3.0

    assert isinstance(rays.L, be.ndarray)
    assert rays.L.shape == (1,)
    assert rays.L.dtype == float
    assert rays.L[0] == 4.0

    assert isinstance(rays.M, be.ndarray)
    assert rays.M.shape == (1,)
    assert rays.M.dtype == float
    assert rays.M[0] == 5.0

    assert isinstance(rays.N, be.ndarray)
    assert rays.N.shape == (1,)
    assert rays.N.dtype == float
    assert rays.N[0] == 6.0

    assert isinstance(rays.i, be.ndarray)
    assert rays.i.shape == (1,)
    assert rays.i.dtype == float
    assert rays.i[0] == 7.0

    assert isinstance(rays.w, be.ndarray)
    assert rays.w.shape == (1,)
    assert rays.w.dtype == float
    assert rays.w[0] == 8.0

    assert isinstance(rays.opd, be.ndarray)
    assert rays.opd.shape == (1,)
    assert rays.opd.dtype == float
    assert rays.opd[0] == 0.0


def test_rotate_x():
    rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    rays.rotate_x(be.pi / 2)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(-3.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(0.0, abs=1e-10)

    rays.rotate_x(-be.pi / 2)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.rotate_x(be.pi)

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

    rays.rotate_y(be.pi / 2)

    assert rays.x[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(-1.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(0.0, abs=1e-10)

    rays.rotate_y(-be.pi / 2)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.rotate_y(be.pi)

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

    rays.rotate_z(be.pi / 2)

    assert rays.x[0] == pytest.approx(-2.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.rotate_z(-be.pi / 2)

    assert rays.x[0] == pytest.approx(1.0, abs=1e-10)
    assert rays.y[0] == pytest.approx(2.0, abs=1e-10)
    assert rays.z[0] == pytest.approx(3.0, abs=1e-10)
    assert rays.L[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.M[0] == pytest.approx(0.0, abs=1e-10)
    assert rays.N[0] == pytest.approx(1.0, abs=1e-10)

    rays.rotate_z(be.pi)

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

    assert isinstance(rays.x, be.ndarray)
    assert rays.x.shape == (1,)
    assert rays.x.dtype == float
    assert rays.x[0] == pytest.approx(0.0, abs=1e-10)

    assert isinstance(rays.y, be.ndarray)
    assert rays.y.shape == (1,)
    assert rays.y.dtype == float
    assert rays.y[0] == pytest.approx(1.0, abs=1e-10)

    assert isinstance(rays.z, be.ndarray)
    assert rays.z.shape == (1,)
    assert rays.z.dtype == float
    assert rays.z[0] == pytest.approx(-10.0, abs=1e-10)

    assert isinstance(rays.u, be.ndarray)
    assert rays.u.shape == (1,)
    assert rays.u.dtype == float
    assert rays.u[0] == pytest.approx(0.1, abs=1e-10)

    assert isinstance(rays.i, be.ndarray)
    assert rays.i.shape == (1,)
    assert rays.i.dtype == float
    assert rays.i[0] == pytest.approx(1.0, abs=1e-10)

    assert isinstance(rays.w, be.ndarray)
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


class TestPolarizationState:
    def test_constructor(self):
        state = PolarizationState(is_polarized=True, Ex=1, Ey=2, phase_x=0, phase_y=1)
        assert state.is_polarized is True
        assert state.Ex == 1 / be.sqrt(5)
        assert state.Ey == 2 / be.sqrt(5)
        assert state.phase_x == 0
        assert state.phase_y == 1

    def test_constructor_invalid(self):
        with pytest.raises(ValueError):
            PolarizationState(is_polarized=True, Ex=1, Ey=2, phase_x=0, phase_y=None)
        with pytest.raises(ValueError):
            PolarizationState(is_polarized=False, Ex=1, Ey=2, phase_x=None, phase_y=1)

    def test_str(self):
        state = PolarizationState(is_polarized=True, Ex=1, Ey=0, phase_x=0, phase_y=1)
        val = "Polarized Light: Ex: 1.0, Ey: 0.0, Phase x: 0.0, Phase y: 1.0"
        assert str(state) == val

    def test_str_unpolarized(self):
        state = PolarizationState(is_polarized=False)
        assert str(state) == "Unpolarized Light"

    def test_repr(self):
        state = PolarizationState(is_polarized=True, Ex=0, Ey=1, phase_x=0, phase_y=1)
        val = "Polarized Light: Ex: 0.0, Ey: 1.0, Phase x: 0.0, Phase y: 1.0"
        assert repr(state) == val


class TestCreatePolarization:
    def test_create_polarization_unpolarized(self):
        state = create_polarization("unpolarized")
        assert state.is_polarized is False
        assert state.Ex is None
        assert state.Ey is None
        assert state.phase_x is None
        assert state.phase_y is None

    def test_create_polarization_horizontal(self):
        state = create_polarization("H")
        assert state.is_polarized is True
        assert state.Ex == pytest.approx(1.0, abs=1e-10)
        assert state.Ey == pytest.approx(0.0, abs=1e-10)
        assert state.phase_x == pytest.approx(0.0, abs=1e-10)
        assert state.phase_y == pytest.approx(0.0, abs=1e-10)

    def test_create_polarization_vertical(self):
        state = create_polarization("V")
        assert state.is_polarized is True
        assert state.Ex == pytest.approx(0.0, abs=1e-10)
        assert state.Ey == pytest.approx(1.0, abs=1e-10)
        assert state.phase_x == pytest.approx(0.0, abs=1e-10)
        assert state.phase_y == pytest.approx(0.0, abs=1e-10)

    def test_create_polarization_linear_45(self):
        state = create_polarization("L+45")
        assert state.is_polarized is True
        assert state.Ex == pytest.approx(be.sqrt(2) / 2, abs=1e-10)
        assert state.Ey == pytest.approx(be.sqrt(2) / 2, abs=1e-10)
        assert state.phase_x == pytest.approx(0.0, abs=1e-10)
        assert state.phase_y == pytest.approx(0.0, abs=1e-10)

    def test_create_polarization_linear_minus_45(self):
        state = create_polarization("L-45")
        assert state.is_polarized is True
        assert state.Ex == pytest.approx(be.sqrt(2) / 2, abs=1e-10)
        assert state.Ey == pytest.approx(-be.sqrt(2) / 2, abs=1e-10)
        assert state.phase_x == pytest.approx(0.0, abs=1e-10)
        assert state.phase_y == pytest.approx(0.0, abs=1e-10)

    def test_create_polarization_right_circular(self):
        state = create_polarization("RCP")
        assert state.is_polarized is True
        assert state.Ex == pytest.approx(be.sqrt(2) / 2, abs=1e-10)
        assert state.Ey == pytest.approx(be.sqrt(2) / 2, abs=1e-10)
        assert state.phase_x == pytest.approx(0.0, abs=1e-10)
        assert state.phase_y == pytest.approx(-be.pi / 2, abs=1e-10)

    def test_create_polarization_left_circular(self):
        state = create_polarization("LCP")
        assert state.is_polarized is True
        assert state.Ex == pytest.approx(be.sqrt(2) / 2, abs=1e-10)
        assert state.Ey == pytest.approx(be.sqrt(2) / 2, abs=1e-10)
        assert state.phase_x == pytest.approx(0.0, abs=1e-10)
        assert state.phase_y == pytest.approx(be.pi / 2, abs=1e-10)

    def test_create_polarization_invalid(self):
        with pytest.raises(ValueError):
            create_polarization("invalid")


class TestPolarizedRays:
    def test_init(self):
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
        assert be.array_equal(rays.p[0], be.eye(3))

    def test_get_output_field(self):
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
        assert be.array_equal(output_field, E)

    def test_update_intensity(self):
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
        assert rays.i[0] == pytest.approx(1.0, abs=1e-10)

        # test case for unpolarized light
        state = PolarizationState(is_polarized=False)
        rays.update_intensity(state)
        assert rays.i.shape == (1,)
        assert rays.i[0] == pytest.approx(1.0, abs=1e-10)

    def test_update(self):
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
        assert be.array_equal(rays.p, jones_matrix)

        # test case when k not orthogonal to N0
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.1])
        rays.N0 = be.sqrt([1 - 0.1**2])
        rays.update(jones_matrix)
        assert rays.p.shape == (1, 3, 3)
        jones_matrix = be.array(
            [[[1.0, 0.0, 0.0], [0.0, 0.99498744, -0.1], [0.0, 0.1, 0.99498744]]],
        )
        assert be.allclose(rays.p, jones_matrix, atol=1e-10)

        # test case when jones = None
        rays.L0 = be.array([0.0])
        rays.M0 = be.array([0.0])
        rays.N0 = be.array([1.0])
        rays.update(None)
        assert rays.p.shape == (1, 3, 3)
        assert be.allclose(rays.p, jones_matrix, atol=1e-10)

    def test_get_3d_electric_field(self):
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
        assert be.allclose(E, be.array([[1.0, 0.0, 0.0]]), atol=1e-10)

    def test_get_3d_electric_field_error(self):
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

        assert be.allclose(rays.x, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert be.allclose(rays.y, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert be.allclose(rays.z, be.array([-0.88839505, -0.88839505]), atol=1e-8)
        assert be.allclose(rays.L, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert be.allclose(rays.M, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert be.allclose(rays.N, be.array([0.96882189, 0.96882189]), atol=1e-8)
        assert be.allclose(rays.i, be.array([1.0, 1.0]), atol=1e-8)
        assert be.allclose(rays.w, be.array([0.55, 0.55]), atol=1e-8)

    def test_generate_rays_telecentric(self):
        lens = UVProjectionLens()
        generator = RayGenerator(lens)

        Hx = 0.0
        Hy = 1.0
        Px = 0.8
        Py = 0.0
        wavelength = 0.248

        rays = generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        assert be.isclose(rays.x[0], 0.0, atol=1e-8)
        assert be.isclose(rays.y[0], 48.0, atol=1e-8)
        assert be.isclose(rays.z[0], -110.85883544, atol=1e-8)
        assert be.isclose(rays.L[0], 0.10674041, atol=1e-8)
        assert be.isclose(rays.M[0], 0.0, atol=1e-8)
        assert be.isclose(rays.N[0], 0.99428692, atol=1e-8)
        assert be.isclose(rays.i[0], 1.0, atol=1e-8)
        assert be.isclose(rays.w[0], 0.248, atol=1e-8)

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

        assert be.allclose(rays.x, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert be.allclose(rays.y, be.array([-0.23535066, -0.1909309]), atol=1e-8)
        assert be.allclose(rays.z, be.array([-0.88839505, -0.88839505]), atol=1e-8)
        assert be.allclose(rays.L, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert be.allclose(rays.M, be.array([0.17519154, 0.17519154]), atol=1e-8)
        assert be.allclose(rays.N, be.array([0.96882189, 0.96882189]), atol=1e-8)
        assert be.allclose(rays.i, be.array([1.0, 1.0]), atol=1e-8)
        assert be.allclose(rays.w, be.array([0.55, 0.55]), atol=1e-8)

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
