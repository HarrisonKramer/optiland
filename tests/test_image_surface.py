import pytest
import numpy as np
from optiland.coordinate_system import CoordinateSystem
from optiland.surfaces.image_surface import ImageSurface
from optiland.materials import IdealMaterial
from optiland.geometries import StandardGeometry
from optiland.rays import ParaxialRays, RealRays


@pytest.fixture
def setup_image_surface():
    cs = CoordinateSystem()
    geometry = StandardGeometry(cs, 100)
    material_pre = IdealMaterial(1, 0)
    material_post = IdealMaterial(1, 0)
    image_surface = ImageSurface(
        geometry=geometry,
        material_pre=material_pre,
        material_post=material_post,
        aperture=None
    )
    return image_surface, geometry, material_pre, material_post


def test_initialization(setup_image_surface):
    image_surface, geometry, material_pre, material_post = setup_image_surface
    assert image_surface.geometry == geometry
    assert image_surface.material_pre == material_pre
    assert image_surface.material_post == material_post
    assert not image_surface.is_stop


def test_trace_paraxial(setup_image_surface):
    image_surface, _, _, _ = setup_image_surface
    y = np.array([1])
    u = np.array([0])
    z = np.array([-10])
    w = np.array([1])
    rays = ParaxialRays(y, u, z, w)
    image_surface._trace_paraxial(rays)


def test_interact(setup_image_surface):
    image_surface, _, _, _ = setup_image_surface
    x = np.random.rand(10)
    rays = RealRays(x, x, x, x, x, x, x, x)
    modified_rays = image_surface._interact(rays)
    assert modified_rays == rays
