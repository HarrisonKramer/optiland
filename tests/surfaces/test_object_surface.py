# tests/surfaces/test_object_surface.py
"""
Tests for the ObjectSurface class in optiland.surfaces.object_surface.
"""
import optiland.backend as be
import pytest
import numpy as np

from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.materials import IdealMaterial
from optiland.rays import ParaxialRays, RealRays
from optiland.surfaces.object_surface import ObjectSurface


@pytest.fixture
def setup_object_surface():
    """
    Sets up a basic ObjectSurface for testing.

    Returns:
        A tuple containing the ObjectSurface instance and its geometry.
    """
    cs = CoordinateSystem()
    geometry = Plane(cs)
    material = IdealMaterial(1, 0)
    object_surface = ObjectSurface(geometry=geometry, material_post=material)
    return object_surface, geometry


def test_is_infinite(set_test_backend, setup_object_surface):
    """
    Tests the `is_infinite` property, which should be True when the object
    surface is at infinity and False otherwise.
    """
    object_surface, geometry = setup_object_surface

    # Test with object at infinity
    geometry.cs.z = -be.array(be.inf)
    assert object_surface.is_infinite

    # Test with finite object distance
    geometry.cs.z = be.array(-100)
    assert not object_surface.is_infinite


def test_set_aperture(set_test_backend, setup_object_surface):
    """
    Smoke test for the `set_aperture` method to ensure it can be called
    without error. The ObjectSurface does not have a conventional aperture.
    """
    object_surface, _ = setup_object_surface
    object_surface.set_aperture()


def test_trace(set_test_backend, setup_object_surface):
    """
    Tests the main `trace` method. For an ObjectSurface, this should simply
    return the rays unmodified.
    """
    object_surface, _ = setup_object_surface
    x = np.random.rand(10)
    rays = RealRays(x, x, x, x, x, x, x, x)
    traced_rays = object_surface.trace(rays)
    assert traced_rays == rays


def test_trace_paraxial(set_test_backend, setup_object_surface):
    """
    Smoke test for the `_trace_paraxial` method. This method is a placeholder
    in the ObjectSurface and should execute without error.
    """
    object_surface, _ = setup_object_surface
    y = be.array([1])
    u = be.array([0])
    z = be.array([-10])
    w = be.array([1])
    rays = ParaxialRays(y, u, z, w)
    object_surface._trace_paraxial(rays)


def test_trace_real(set_test_backend, setup_object_surface):
    """
    Smoke test for the `_trace_real` method. This method is a placeholder
    in the ObjectSurface and should execute without error.
    """
    object_surface, _ = setup_object_surface
    x = be.random_uniform(size=10)
    rays = RealRays(x, x, x, x, x, x, x, x)
    object_surface._trace_real(rays)


def test_interact(set_test_backend, setup_object_surface):
    """
    Tests the `_interact` method. The ObjectSurface does not interact with
    rays, so it should return them unmodified.
    """
    object_surface, _ = setup_object_surface
    x = be.random_uniform(size=10)
    rays = RealRays(x, x, x, x, x, x, x, x)
    interacted_rays = object_surface._interact(rays)
    assert interacted_rays == rays