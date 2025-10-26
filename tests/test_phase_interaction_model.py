
import pytest
from unittest.mock import Mock, PropertyMock
from optiland import backend as be
from .utils import assert_allclose
from optiland.optic.optic import Optic
from optiland.geometries.plane import Plane
from optiland.materials.ideal import IdealMaterial
from optiland.phase.radial import RadialPhaseProfile
from optiland.interactions.phase_interaction_model import PhaseInteractionModel
from optiland.rays.real_rays import RealRays
from optiland.rays.paraxial_rays import ParaxialRays
from optiland.surfaces.standard_surface import Surface
from optiland.coordinate_system import CoordinateSystem

@pytest.fixture
def mock_surface(set_test_backend):
    # Create a mock surface with the necessary attributes
    surface = Mock(spec=Surface)
    surface.geometry = Plane(CoordinateSystem())
    surface.material_pre = IdealMaterial(n=1.0)
    surface.material_post = IdealMaterial(n=1.5)
    # Mock methods that might be called
    surface.geometry.surface_normal = Mock(return_value=(be.zeros(1), be.zeros(1), be.ones(1)))
    return surface

def test_phase_interaction_model_init(mock_surface):
    phase_profile = RadialPhaseProfile(coefficients=[0.1])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)
    assert model.phase_profile == phase_profile
    assert model.parent_surface == mock_surface

def test_phase_interaction_model_init_raises_type_error(mock_surface):
    # Test that it raises an error if not a PlaneGeometry
    mock_surface.geometry = Mock() # Not a Plane
    phase_profile = RadialPhaseProfile(coefficients=[0.1])
    with pytest.raises(TypeError):
        PhaseInteractionModel(mock_surface, phase_profile)

def test_interact_real_rays_transmission(mock_surface):
    # Metalens focusing at f=100mm
    # phi = -k0/(2f) * r^2
    f = 100.0
    w = 0.5
    k0 = 2 * be.pi / w
    coeff = -k0 / (2*f)
    phase_profile = RadialPhaseProfile(coefficients=[coeff])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    rays = RealRays(x=be.array([1.0]), y=be.array([0.0]), z=be.array([0.0]),
                    L=be.array([0.0]), M=be.array([0.0]), N=be.array([1.0]),
                    wavelength=w, intensity=be.array([1.0]))

    # Expected outgoing angle: L = -x / (n2 * f)
    expected_L = -1.0 / (mock_surface.material_post.n(w) * f)

    interacted_rays = model.interact_real_rays(rays)

    assert_allclose(interacted_rays.L, be.array([expected_L]), atol=1e-6)
    assert_allclose(interacted_rays.M, be.array([0.0]), atol=1e-6)

def test_interact_real_rays_reflection(mock_surface):
    f = 100.0
    w = 0.5
    k0 = 2 * be.pi / w
    coeff = -k0 / (2*f) # Same phase profile, different interaction
    phase_profile = RadialPhaseProfile(coefficients=[coeff])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=True)

    rays = RealRays(x=be.array([1.0]), y=be.array([0.0]), z=be.array([0.0]),
                    L=be.array([0.0]), M=be.array([0.0]), N=be.array([1.0]),
                    wavelength=w, intensity=be.array([1.0]))

    # Expected outgoing angle for reflection is more complex, but N should be negative
    interacted_rays = model.interact_real_rays(rays)
    assert be.all(interacted_rays.N < 0)

def test_interact_paraxial_rays(mock_surface):
    f = 100.0
    w = 0.5
    k0 = 2 * be.pi / w
    coeff = -k0 / (2*f)
    phase_profile = RadialPhaseProfile(coefficients=[coeff])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    rays = ParaxialRays(y=be.array([1.0]), u=be.array([0.0]), z=be.array([0.0]), wavelength=w)

    # Expected paraxial angle: u_out = u_in - y/f (for n2=1)
    # With n2 != 1, u_out = u_in/n2 - y/(n2*f)
    n2 = mock_surface.material_post.n(w)
    expected_u = -1.0 / (n2 * f)

    interacted_rays = model.interact_paraxial_rays(rays)

    assert_allclose(interacted_rays.u, be.array([expected_u]), atol=1e-6)

def test_tir_case(mock_surface):
    # Create a gradient so large it must cause TIR
    phase_profile = RadialPhaseProfile(coefficients=[1e6])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    rays = RealRays(x=be.array([1.0]), y=be.array([0.0]), z=be.array([0.0]),
                    L=be.array([0.0]), M=be.array([0.0]), N=be.array([1.0]),
                    wavelength=0.5, intensity=be.array([1.0]))

    interacted_rays = model.interact_real_rays(rays)
    # The ray should be clipped (intensity becomes zero)
    assert be.all(interacted_rays.i == 0)

def test_serialization(mock_surface):
    phase_profile = RadialPhaseProfile(coefficients=[0.1, 0.2])
    model = PhaseInteractionModel(mock_surface, phase_profile, is_reflective=False)

    data = model.to_dict()
    assert data["type"] == "PhaseInteractionModel"
    assert data["phase_profile"]["phase_type"] == "radial"

    new_model = PhaseInteractionModel.from_dict(data, mock_surface)
    assert isinstance(new_model, PhaseInteractionModel)
    assert isinstance(new_model.phase_profile, RadialPhaseProfile)
    assert new_model.phase_profile.coefficients == [0.1, 0.2]
