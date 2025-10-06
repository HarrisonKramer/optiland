# tests/surfaces/test_image_surface.py
"""
Tests for the ImageSurface class in optiland.surfaces.image_surface.
"""
import optiland.backend as be
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import StandardGeometry
from optiland.materials import IdealMaterial
from optiland.rays import ParaxialRays, RealRays
from optiland.surfaces.image_surface import ImageSurface


@pytest.fixture
def setup_image_surface():
    """
    Sets up a basic ImageSurface for testing.

    Returns:
        A tuple containing the ImageSurface instance, its geometry, and its
        pre- and post-materials.
    """
    cs = CoordinateSystem()
    geometry = StandardGeometry(cs, 100)
    material_pre = IdealMaterial(1, 0)
    material_post = IdealMaterial(1, 0)
    image_surface = ImageSurface(
        geometry=geometry,
        material_pre=material_pre,
        material_post=material_post,
        aperture=None,
    )
    return image_surface, geometry, material_pre, material_post


def test_initialization(set_test_backend, setup_image_surface):
    """
    Tests that the ImageSurface is initialized with the correct attributes.
    """
    image_surface, geometry, material_pre, material_post = setup_image_surface
    assert image_surface.geometry == geometry
    assert image_surface.material_pre == material_pre
    assert image_surface.material_post == material_post
    assert not image_surface.is_stop


def test_trace_paraxial(set_test_backend, setup_image_surface):
    """
    Tests the paraxial ray trace through an ImageSurface.

    This is primarily a smoke test to ensure the method executes without error,
    as the ImageSurface does not modify paraxial rays but simply propagates
    them to the surface.
    """
    image_surface, _, _, _ = setup_image_surface
    y = be.array([1])
    u = be.array([0])
    z = be.array([-10])
    w = be.array([1])
    rays = ParaxialRays(y, u, z, w)
    image_surface._trace_paraxial(rays)


def test_interact(set_test_backend, setup_image_surface):
    """
    Tests the interaction of real rays with an ImageSurface.

    The ImageSurface should not modify the rays, so this test verifies that
    the output rays are identical to the input rays.
    """
    image_surface, _, _, _ = setup_image_surface
    x = be.random_uniform(size=10)
    rays = RealRays(x, x, x, x, x, x, x, x)
    modified_rays = image_surface._interact(rays)
    assert modified_rays == rays