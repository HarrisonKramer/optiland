import pytest
import abc
from unittest.mock import MagicMock, patch

# Assuming 'be' is accessible similarly to how it's used in the main code
# If not, direct mocking of be.logging might be needed.
# import optiland.backend as be
# For now, let's assume direct patching of logging if be.logging is tricky to get here
# from optiland.backend import be # This might cause issues if backend initializes complex things

from optiland.solves.base import BaseSolve
from optiland.solves.curvature_solve_base import CurvatureSolveBase
from optiland.solves.marginal_ray_angle_solve import MarginalRayAngleSolve
from optiland.solves.chief_ray_angle_solve import ChiefRayAngleSolve

# Mock Optic and related objects for testing
# These mocks will need to be adjusted based on actual interactions in the methods.

class MockOptic:
    def __init__(self):
        self.surface_group = MagicMock()
        self.paraxial = MagicMock()
        self.wavelengths = MagicMock()
        # Mocking primary wavelength value directly for simplicity in tests
        self.wavelengths.primary_wavelength_value = 0.58756 # Example primary wavelength
        self.object_space_material = None # Default

        # For refractive index testing via paraxial object
        self.paraxial.indices_n = []
        self.paraxial.indices_n_prime = []


class MockSurface:
    def __init__(self):
        self.geometry = MagicMock()
        self.geometry.curvature = 0.0
        self.material_before = None
        self.material_after = None

class MockMaterial:
    def __init__(self, index_val):
        self.index_val = index_val
    def get_index(self, wavelength):
        # Simple mock: returns a fixed index regardless of wavelength for testing
        return self.index_val


# A concrete implementation of CurvatureSolveBase for testing its non-abstract methods
class ConcreteCurvatureSolve(CurvatureSolveBase):
    def __init__(self, optic, surface_idx, angle, mock_paraxial_data_return=None):
        super().__init__(optic, surface_idx, angle)
        self._mock_paraxial_data_return = mock_paraxial_data_return

    def _get_paraxial_data_at_surface(self):
        if self._mock_paraxial_data_return is not None:
            if isinstance(self._mock_paraxial_data_return, Exception):
                raise self._mock_paraxial_data_return
            return self._mock_paraxial_data_return
        # Default dummy implementation, should be overridden by mock_paraxial_data_return
        # or test should focus on methods not calling this.
        return (1.0, 0.1, 1.0, 1.5) # y_k, u_k, n_k, n_prime_k


@pytest.fixture
def mock_optic():
    optic = MockOptic()
    # Setup surfaces
    num_surfaces = 3
    optic.surface_group.surfaces = [MockSurface() for _ in range(num_surfaces)]

    # Setup paraxial refractive indices (example)
    optic.paraxial.indices_n = [1.0, 1.5, 1.6]
    optic.paraxial.indices_n_prime = [1.5, 1.6, 1.0]
    return optic

# --- Tests for CurvatureSolveBase ---

def test_curvature_solve_base_init(mock_optic):
    solve = ConcreteCurvatureSolve(mock_optic, 1, 0.05)
    assert solve.optic == mock_optic
    assert solve.surface_idx == 1
    assert solve.angle == 0.05

def test_curvature_solve_base_to_dict(mock_optic):
    solve = ConcreteCurvatureSolve(mock_optic, 1, 0.05)
    expected_dict = {
        "type": "ConcreteCurvatureSolve", # Relies on BaseSolve._registry mechanism
        "surface_idx": 1,
        "angle": 0.05,
    }
    # Mocking super().to_dict() which comes from BaseSolve
    with patch.object(BaseSolve, 'to_dict', return_value={"type": "ConcreteCurvatureSolve"}):
        d = solve.to_dict()
    assert d == expected_dict

def test_curvature_solve_base_from_dict_abstract_error(mock_optic):
    data = {"type": "CurvatureSolveBase", "surface_idx": 0, "angle": 0.1}
    with pytest.raises(TypeError, match="CurvatureSolveBase is an abstract class"):
        CurvatureSolveBase.from_dict(mock_optic, data)

def test_concrete_curvature_solve_from_dict(mock_optic):
    # This implicitly tests CurvatureSolveBase.from_dict if ConcreteCurvatureSolve doesn't override it
    # and if ConcreteCurvatureSolve is registered (which it is by inheriting BaseSolve)
    data = {"type": "ConcreteCurvatureSolve", "surface_idx": 1, "angle": 0.05}
    solve = ConcreteCurvatureSolve.from_dict(mock_optic, data)
    assert isinstance(solve, ConcreteCurvatureSolve)
    assert solve.optic == mock_optic
    assert solve.surface_idx == 1
    assert solve.angle == 0.05

def test_curvature_solve_base_from_dict_missing_keys(mock_optic):
    with pytest.raises(ValueError, match="must include 'surface_idx' and 'angle'"):
        ConcreteCurvatureSolve.from_dict(mock_optic, {"type": "ConcreteCurvatureSolve", "surface_idx": 1})
    with pytest.raises(ValueError, match="must include 'surface_idx' and 'angle'"):
        ConcreteCurvatureSolve.from_dict(mock_optic, {"type": "ConcreteCurvatureSolve", "angle": 0.1})


# --- Tests for CurvatureSolveBase.apply() via ConcreteCurvatureSolve ---
# These tests examine the core curvature calculation logic and edge cases

def test_apply_curvature_nominal_case(mock_optic):
    # y_k=10, u_k=0.0, n_k=1.0, n_prime_k=1.5, target_angle u_prime_k_target=-0.05
    # c_k = (1.0 * 0.0 - 1.5 * -0.05) / (10 * (1.5 - 1.0))
    # c_k = (0.075) / (10 * 0.5) = 0.075 / 5.0 = 0.015
    paraxial_data = (10.0, 0.0, 1.0, 1.5)
    solve = ConcreteCurvatureSolve(mock_optic, 1, -0.05, mock_paraxial_data_return=paraxial_data)
    solve.apply()
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == pytest.approx(0.015)

def test_apply_curvature_y_k_zero_cannot_achieve(mock_optic):
    paraxial_data = (0.0, 0.1, 1.0, 1.5) # y_k = 0
    solve = ConcreteCurvatureSolve(mock_optic, 1, -0.05, mock_paraxial_data_return=paraxial_data)
    original_curvature = mock_optic.surface_group.surfaces[1].geometry.curvature
    solve.apply()
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == original_curvature

def test_apply_curvature_y_k_zero_already_achieved(mock_optic):
    # n_k*u_k = 1.0 * 0.02 = 0.02
    # n_prime_k*u_prime_k_target = 1.5 * (0.02/1.5) = 0.02
    # So, n_k*u_k == n_prime_k*u_prime_k_target
    paraxial_data = (0.0, 0.02, 1.0, 1.5)
    solve = ConcreteCurvatureSolve(mock_optic, 1, 0.02/1.5, mock_paraxial_data_return=paraxial_data)
    original_curvature = mock_optic.surface_group.surfaces[1].geometry.curvature
    solve.apply()
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == original_curvature


def test_apply_curvature_indices_equal_cannot_achieve(mock_optic):
    paraxial_data = (10.0, 0.1, 1.5, 1.5) # n_k = n_prime_k
    solve = ConcreteCurvatureSolve(mock_optic, 1, 0.05, mock_paraxial_data_return=paraxial_data) # target different
    original_curvature = mock_optic.surface_group.surfaces[1].geometry.curvature
    solve.apply()
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == original_curvature

def test_apply_curvature_indices_equal_already_achieved(mock_optic):
    # n_k*u_k = 1.5 * 0.1 = 0.15
    # n_prime_k*u_prime_k_target = 1.5 * 0.1 = 0.15
    paraxial_data = (10.0, 0.1, 1.5, 1.5) # n_k = n_prime_k
    solve = ConcreteCurvatureSolve(mock_optic, 1, 0.1, mock_paraxial_data_return=paraxial_data) # target same effect
    original_curvature = mock_optic.surface_group.surfaces[1].geometry.curvature
    solve.apply()
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == original_curvature


def test_apply_curvature_denominator_zero(mock_optic):
    # y_k is not zero, but (n_prime_k - n_k) is extremely small, making denominator effectively zero.
    # This test is for the y_k * (n_prime_k - n_k) denominator check
    paraxial_data = (10.0, 0.1, 1.0, 1.0000000000001) # n_prime_k - n_k is tiny
    solve = ConcreteCurvatureSolve(mock_optic, 1, -0.05, mock_paraxial_data_return=paraxial_data)
    original_curvature = mock_optic.surface_group.surfaces[1].geometry.curvature
    solve.apply()
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == original_curvature

def test_apply_curvature_surface_idx_out_of_bounds_for_surfaces(mock_optic):
    paraxial_data = (10.0, 0.0, 1.0, 1.5)
    solve = ConcreteCurvatureSolve(mock_optic, 99, -0.05, mock_paraxial_data_return=paraxial_data) # Invalid surface_idx
    # We expect apply to return early without changing curvature or erroring
    original_curvature = mock_optic.surface_group.surfaces[0].geometry.curvature # Check any surface
    solve.apply()
    # Ensure no curvature was changed on any existing surface
    for surface in mock_optic.surface_group.surfaces:
        assert surface.geometry.curvature == 0.0 # Assuming they all start at 0.0


def test_apply_curvature_get_paraxial_data_raises_index_error(mock_optic):
    solve = ConcreteCurvatureSolve(mock_optic, 1, -0.05, mock_paraxial_data_return=IndexError("Test Ray Data OOB"))
    original_curvature = mock_optic.surface_group.surfaces[1].geometry.curvature
    solve.apply() # Should return silently
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == original_curvature

def test_apply_curvature_get_paraxial_data_raises_general_error(mock_optic):
    solve = ConcreteCurvatureSolve(mock_optic, 1, -0.05, mock_paraxial_data_return=ValueError("Test General Error"))
    original_curvature = mock_optic.surface_group.surfaces[1].geometry.curvature
    solve.apply() # Should return silently
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == original_curvature


# --- Tests for MarginalRayAngleSolve ---

@pytest.fixture
def mock_marginal_ray_solve(mock_optic):
    # Setup marginal ray data
    mock_optic.paraxial.marginal_ray.return_value = ([0.0, 5.0, 4.8], [0.2, 0.18, 0.15]) # y_values, u_values
    return MarginalRayAngleSolve(mock_optic, 1, -0.05) # Target surface 1, angle -0.05

def test_marginal_ray_angle_solve_init(mock_marginal_ray_solve, mock_optic):
    assert mock_marginal_ray_solve.optic == mock_optic
    assert mock_marginal_ray_solve.surface_idx == 1
    assert mock_marginal_ray_solve.angle == -0.05

def test_marginal_ray_get_paraxial_data(mock_marginal_ray_solve, mock_optic):
    y_k, u_k, n_k, n_prime_k = mock_marginal_ray_solve._get_paraxial_data_at_surface()
    assert y_k == 5.0  # y_values[1]
    assert u_k == 0.18 # u_values[1]
    assert n_k == mock_optic.paraxial.indices_n[1] # 1.5
    assert n_prime_k == mock_optic.paraxial.indices_n_prime[1] # 1.6
    assert type(y_k) is float
    assert type(u_k) is float
    assert type(n_k) is float
    assert type(n_prime_k) is float


def test_marginal_ray_angle_solve_apply(mock_marginal_ray_solve, mock_optic):
    # Uses y=5, u=0.18, n=1.5, n_prime=1.6 (from mock_optic.paraxial.indices_n/n_prime at index 1)
    # Target angle = -0.05
    # c_k = (1.5 * 0.18 - 1.6 * -0.05) / (5.0 * (1.6 - 1.5))
    # c_k = (0.27 - (-0.08)) / (5.0 * 0.1)
    # c_k = (0.35) / (0.5) = 0.7
    mock_marginal_ray_solve.apply()
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == pytest.approx(0.7)

def test_marginal_ray_get_paraxial_data_index_error_ray_data(mock_optic):
    mock_optic.paraxial.marginal_ray.return_value = ([0.0, 5.0], [0.2, 0.18]) # Data only for 2 surfaces
    solve = MarginalRayAngleSolve(mock_optic, 2, -0.05) # Requesting index 2
    with pytest.raises(IndexError, match="out of bounds for marginal ray data"):
        solve._get_paraxial_data_at_surface()

def test_marginal_ray_get_paraxial_data_index_error_indices_n(mock_optic):
    mock_optic.paraxial.marginal_ray.return_value = ([0.0, 5.0, 4.8], [0.2, 0.18, 0.15])
    mock_optic.paraxial.indices_n = [1.0, 1.5] # Not enough data
    solve = MarginalRayAngleSolve(mock_optic, 2, -0.05)
    with pytest.raises(IndexError, match="out of bounds for paraxial refractive index arrays"):
        solve._get_paraxial_data_at_surface()

def test_marginal_ray_get_paraxial_data_attribute_error_paraxial(mock_optic):
    mock_optic.paraxial = None
    solve = MarginalRayAngleSolve(mock_optic, 1, -0.05)
    with pytest.raises(AttributeError, match="Paraxial analysis data is not available"):
        solve._get_paraxial_data_at_surface()

def test_marginal_ray_get_paraxial_data_fallback_indices_not_implemented(mock_optic):
    # Test the fallback path for indices if paraxial.indices_n/n_prime don't exist
    mock_optic.paraxial.marginal_ray.return_value = ([0.0, 5.0, 4.8], [0.2, 0.18, 0.15])
    # Must use try-del for attributes that might not exist if class structure changes
    try: del mock_optic.paraxial.indices_n
    except AttributeError: pass
    try: del mock_optic.paraxial.indices_n_prime
    except AttributeError: pass

    # Setup materials for fallback
    mock_optic.surface_group.surfaces[0].material_before = MockMaterial(1.0) # Object space
    mock_optic.surface_group.surfaces[0].material_after = MockMaterial(1.51)
    mock_optic.surface_group.surfaces[1].material_after = MockMaterial(1.61) # n_prime for surface 1

    solve = MarginalRayAngleSolve(mock_optic, 1, -0.05) # Test surface 1

    # This test is tricky because the fallback has a NotImplementedError
    # Let's assume the fallback logic is more complete for a moment, or mock it out
    # For now, let's specifically test that if the preferred attributes are missing,
    # and if the fallback logic for material_after on prev_surface is missing, it raises ValueError

    # To test the NotImplementedError path in the original code for surface_idx > 0
    # where current_surface.material_after is None (and thus n_prime_k fails)
    mock_optic.surface_group.surfaces[1].material_after = None # This will cause NotImplementedError for n_prime_k
    # And ensure prev_surface.material_after exists for n_k
    mock_optic.surface_group.surfaces[0].material_after = MockMaterial(1.51)


    with pytest.raises(NotImplementedError, match="Refractive index determination logic needs to be confirmed"):
        solve._get_paraxial_data_at_surface()


def test_marginal_ray_angle_solve_from_dict(mock_optic):
    data = {"type": "MarginalRayAngleSolve", "surface_idx": 0, "angle": 0.1}
    # Ensure MarginalRayAngleSolve is registered
    assert "MarginalRayAngleSolve" in BaseSolve._registry
    solve = BaseSolve.from_dict(mock_optic, data) # Test via BaseSolve factory
    assert isinstance(solve, MarginalRayAngleSolve)
    assert solve.optic == mock_optic
    assert solve.surface_idx == 0
    assert solve.angle == 0.1


# --- Tests for ChiefRayAngleSolve ---

@pytest.fixture
def mock_chief_ray_solve(mock_optic):
    # Setup chief ray data
    mock_optic.paraxial.chief_ray.return_value = ([0.0, -2.0, -1.8], [0.05, 0.04, 0.03]) # ybar_values, ubar_values
    # Indices from mock_optic: n_k=1.5, n_prime_k=1.6 for surface_idx=1
    return ChiefRayAngleSolve(mock_optic, 1, 0.02) # Target surface 1, angle 0.02

def test_chief_ray_angle_solve_init(mock_chief_ray_solve, mock_optic):
    assert mock_chief_ray_solve.optic == mock_optic
    assert mock_chief_ray_solve.surface_idx == 1
    assert mock_chief_ray_solve.angle == 0.02

def test_chief_ray_get_paraxial_data(mock_chief_ray_solve, mock_optic):
    y_k, u_k, n_k, n_prime_k = mock_chief_ray_solve._get_paraxial_data_at_surface()
    assert y_k == -2.0  # ybar_values[1]
    assert u_k == 0.04  # ubar_values[1]
    assert n_k == mock_optic.paraxial.indices_n[1] # 1.5
    assert n_prime_k == mock_optic.paraxial.indices_n_prime[1] # 1.6
    assert type(y_k) is float
    assert type(u_k) is float
    assert type(n_k) is float
    assert type(n_prime_k) is float


def test_chief_ray_angle_solve_apply(mock_chief_ray_solve, mock_optic):
    # Uses ybar=-2, ubar=0.04, n=1.5, n_prime=1.6 (from mock_optic.paraxial.indices_n/n_prime at index 1)
    # Target angle = 0.02
    # c_k = (1.5 * 0.04 - 1.6 * 0.02) / (-2.0 * (1.6 - 1.5))
    # c_k = (0.06 - 0.032) / (-2.0 * 0.1)
    # c_k = (0.028) / (-0.2) = -0.14
    mock_chief_ray_solve.apply()
    assert mock_optic.surface_group.surfaces[1].geometry.curvature == pytest.approx(-0.14)

def test_chief_ray_angle_solve_from_dict(mock_optic):
    data = {"type": "ChiefRayAngleSolve", "surface_idx": 2, "angle": -0.01}
    assert "ChiefRayAngleSolve" in BaseSolve._registry
    solve = BaseSolve.from_dict(mock_optic, data)
    assert isinstance(solve, ChiefRayAngleSolve)
    assert solve.optic == mock_optic
    assert solve.surface_idx == 2
    assert solve.angle == -0.01

# It would also be good to test the fallback index calculation for _get_paraxial_data_at_surface
# more thoroughly if that path is considered critical, by setting up materials on surfaces
# and removing optic.paraxial.indices_n/indices_n_prime. The current test
# `test_marginal_ray_get_paraxial_data_fallback_indices_not_implemented` covers the NotImplementedError.

```
