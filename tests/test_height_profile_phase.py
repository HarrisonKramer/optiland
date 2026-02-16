import pytest
import numpy as np
from optiland import backend as be
from optiland.phase.height_profile import HeightProfile
from optiland.materials.ideal import IdealMaterial
from .utils import assert_allclose

pytest.importorskip("scipy")

@pytest.fixture
def height_data():
    x = be.linspace(-1, 1, 5)
    y = be.linspace(-1, 1, 4)
    height_map = be.array([[i + j for i in x] for j in y])
    material = IdealMaterial(n=1.5)
    return x, y, height_map, material

def test_height_profile_init(height_data):
    x, y, height_map, material = height_data
    profile = HeightProfile(x, y, height_map, material)
    assert profile.x_coords.shape[0] == len(x)
    assert profile.y_coords.shape[0] == len(y)
    assert profile.height_map.shape == (len(y), len(x))
    assert profile.material is material

def test_height_profile_get_phase(height_data):
    x, y, height_map, material = height_data
    profile = HeightProfile(x, y, height_map, material)
    phase = profile.get_phase(be.array([0.0]), be.array([0.0]), be.array([1.0]))
    assert phase.shape == (1,)
    assert isinstance(phase.item(), float)

def test_height_profile_get_gradient(height_data):
    x, y, height_map, material = height_data
    profile = HeightProfile(x, y, height_map, material)
    grad_x, grad_y, grad_z = profile.get_gradient(be.array([0.0]), be.array([0.0]), be.array([1.0]))
    assert grad_x.shape == grad_y.shape == grad_z.shape
    assert_allclose(grad_z, be.zeros_like(grad_z))

def test_height_profile_get_paraxial_gradient(height_data):
    x, y, height_map, material = height_data
    profile = HeightProfile(x, y, height_map, material)
    paraxial = profile.get_paraxial_gradient(be.array([0.0, 0.5]), be.array([1.0]))
    assert paraxial.shape[0] == 2

def test_height_profile_to_from_dict(height_data):
    x, y, height_map, material = height_data
    profile = HeightProfile(x, y, height_map, material)
    data = profile.to_dict()
    new_profile = HeightProfile.from_dict(data)
    assert isinstance(new_profile, HeightProfile)
    assert_allclose(new_profile.x_coords, x)
    assert_allclose(new_profile.y_coords, y)
    assert_allclose(new_profile.height_map, height_map)
