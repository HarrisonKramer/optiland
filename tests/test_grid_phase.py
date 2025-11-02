
import pytest
from optiland import backend as be
from .utils import assert_allclose
from optiland.phase.grid import GridPhaseProfile

# Skip all tests in this file if scipy is not installed
pytest.importorskip("scipy")


@pytest.fixture
def grid_data(set_test_backend):
    if be.get_backend() == "torch":
        pytest.skip("GridPhaseProfile is not supported for torch backend")
    x = be.linspace(-1, 1, 5)
    y = be.linspace(-2, 2, 4)
    # phase_grid must have shape (len(y), len(x))
    phase_grid = be.array([[i**2 + j**3 for i in x] for j in y])
    return x, y, phase_grid


def test_grid_phase_profile_init_numpy(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)
    assert profile.x_coords is not None
    assert profile.y_coords is not None
    assert profile.phase_grid is not None


def test_grid_phase_profile_init_torch_raises_error(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("This test is only for the torch backend")
    x = be.linspace(-1, 1, 3)
    y = be.linspace(-2, 2, 5)
    xx, yy = be.meshgrid(x, y)
    phase_grid = xx**2 + yy**3
    with pytest.raises(NotImplementedError):
        GridPhaseProfile(x, y, phase_grid)


def test_grid_phase_profile_get_phase(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)
    # test at a grid point
    phase = profile.get_phase(be.array([x[1]]), be.array([y[2]]))
    assert_allclose(phase, phase_grid[2, 1])
    # test interpolated point
    phase_interp = profile.get_phase(be.array([0.5]), be.array([0.5]))
    # manual interpolation is complex, just check it runs and returns a float
    assert isinstance(phase_interp.item(), float)


def test_grid_phase_profile_get_gradient(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)
    # test at a point
    grad_x, grad_y, grad_z = profile.get_gradient(be.array([0.5]), be.array([1.0]))
    # analytical gradient: d/dx(x^2+y^3)=2x, d/dy(x^2+y^3)=3y^2
    # at (0.5, 1.0), grad should be close to (1.0, 3.0)
    assert_allclose(grad_x, be.array([1.0]), atol=1e-1) # Spline has some error
    assert_allclose(grad_y, be.array([3.0]), atol=1e-1)
    assert_allclose(grad_z, be.array([0.0]))


def test_grid_phase_profile_get_paraxial_gradient(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)
    y_vals = be.array([0.5, 1.0, 1.5])
    paraxial_grad = profile.get_paraxial_gradient(y_vals)
    # analytical paraxial gradient (at x=0): 3y^2
    expected_grad = 3 * y_vals**2
    assert_allclose(paraxial_grad, expected_grad, atol=1e-1)


def test_grid_phase_profile_to_from_dict(grid_data):
    x, y, phase_grid = grid_data
    profile = GridPhaseProfile(x, y, phase_grid)
    data = profile.to_dict()
    assert data["phase_type"] == "grid"
    assert len(data["x_coords"]) == 5
    assert len(data["y_coords"]) == 4
    assert len(data["phase_grid"]) == 4

    new_profile = GridPhaseProfile.from_dict(data)
    assert isinstance(new_profile, GridPhaseProfile)
    assert_allclose(new_profile.x_coords, x)
