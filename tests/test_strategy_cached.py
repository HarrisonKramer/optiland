import pytest
import optiland.backend as be
from unittest.mock import MagicMock

from optiland.optic.optic import Optic
from optiland.aiming.strategies import CachedAimingStrategy, ParaxialAimingStrategy
from optiland.rays.real_rays import RealRays

@pytest.fixture
def single_thin_lens_optic():
    """A simple optic with a single thin lens."""
    optic = Optic()
    optic.set_field_type("object_height")
    optic.add_field(y=1)
    optic.set_aperture("EPD", 10)
    optic.add_wavelength(value=0.55, is_primary=True)

    optic.add_surface(index=0, surface_type="standard", thickness=100)
    optic.add_surface(index=1, surface_type="paraxial", f=50, is_stop=True, thickness=100)
    optic.add_surface(index=2, surface_type="standard")
    optic.update()
    return optic

def test_cache_miss_and_hit(single_thin_lens_optic):
    """Test that a cache miss calls the wrapped strategy and a hit does not."""
    wrapped_strategy = MagicMock(spec=ParaxialAimingStrategy())
    # Configure the mock to return a valid RealRays object
    dummy_ray = RealRays(x=be.array([0]), y=be.array([0]), z=be.array([0]),
                         L=be.array([0]), M=be.array([0]), N=be.array([1]),
                         intensity=be.array([1]), wavelength=be.array([0.55]))
    wrapped_strategy.aim.return_value = dummy_ray

    cached_strategy = CachedAimingStrategy(wrapped_strategy)
    optic = single_thin_lens_optic
    params = {"optic": optic, "Hx": 0.0, "Hy": 0.1, "Px": 0.0, "Py": 0.1, "wavelength": 0.55}

    # First call - should be a miss
    cached_strategy.aim(**params)
    wrapped_strategy.aim.assert_called_once_with(**params)

    # Second call - should be a hit
    cached_strategy.aim(**params)
    wrapped_strategy.aim.assert_called_once()  # Should not be called again

def test_optic_change_invalidation(single_thin_lens_optic):
    """Test that changing the optic invalidates the cache."""
    wrapped_strategy = MagicMock(spec=ParaxialAimingStrategy())
    dummy_ray = RealRays(x=be.array([0]), y=be.array([0]), z=be.array([0]),
                         L=be.array([0]), M=be.array([0]), N=be.array([1]),
                         intensity=be.array([1]), wavelength=be.array([0.55]))
    wrapped_strategy.aim.return_value = dummy_ray

    cached_strategy = CachedAimingStrategy(wrapped_strategy)
    optic = single_thin_lens_optic
    params = {"optic": optic, "Hx": 0.0, "Hy": 0.1, "Px": 0.0, "Py": 0.1, "wavelength": 0.55}

    # First call
    cached_strategy.aim(**params)
    wrapped_strategy.aim.assert_called_once()

    # Modify the optic
    optic.set_thickness(110, 0)
    params["optic"] = optic # update params with modified optic

    # Second call with modified optic - should be a miss
    cached_strategy.aim(**params)
    assert wrapped_strategy.aim.call_count == 2

def test_cache_eviction(single_thin_lens_optic):
    """Test LRU cache eviction."""
    wrapped_strategy = MagicMock(spec=ParaxialAimingStrategy())
    dummy_ray = RealRays(x=be.array([0]), y=be.array([0]), z=be.array([0]),
                         L=be.array([0]), M=be.array([0]), N=be.array([1]),
                         intensity=be.array([1]), wavelength=be.array([0.55]))
    wrapped_strategy.aim.return_value = dummy_ray

    cached_strategy = CachedAimingStrategy(wrapped_strategy, max_size=2)
    optic = single_thin_lens_optic

    params1 = {"optic": optic, "Hx": 0.0, "Hy": 0.1, "Px": 0.0, "Py": 0.1, "wavelength": 0.55}
    params2 = {"optic": optic, "Hx": 0.0, "Hy": 0.2, "Px": 0.0, "Py": 0.2, "wavelength": 0.55}
    params3 = {"optic": optic, "Hx": 0.0, "Hy": 0.3, "Px": 0.0, "Py": 0.3, "wavelength": 0.55}

    # Fill the cache
    cached_strategy.aim(**params1)
    cached_strategy.aim(**params2)
    assert wrapped_strategy.aim.call_count == 2

    # This should evict params1
    cached_strategy.aim(**params3)
    assert wrapped_strategy.aim.call_count == 3

    # Access params2 to make it most recently used
    cached_strategy.aim(**params2)
    assert wrapped_strategy.aim.call_count == 3

    # This call should be a miss because params1 was evicted
    cached_strategy.aim(**params1)
    assert wrapped_strategy.aim.call_count == 4

def test_vectorized_input(single_thin_lens_optic):
    """Test that vectorized input is handled correctly."""
    paraxial_strategy = ParaxialAimingStrategy()
    cached_strategy = CachedAimingStrategy(paraxial_strategy)
    optic = single_thin_lens_optic

    Hx = be.array([0.0, 0.1])
    Hy = be.array([0.1, 0.2])
    Px = be.array([0.1, 0.2])
    Py = be.array([0.2, 0.3])
    wavelength = 0.55

    # Get expected results from the underlying strategy
    expected_rays = paraxial_strategy.aim(optic, Hx, Hy, Px, Py, wavelength)

    # Get results from the cached strategy
    cached_rays = cached_strategy.aim(optic, Hx, Hy, Px, Py, wavelength)

    assert be.allclose(cached_rays.x, expected_rays.x)
    assert be.allclose(cached_rays.y, expected_rays.y)
    assert be.allclose(cached_rays.z, expected_rays.z)
    assert be.allclose(cached_rays.L, expected_rays.L)
    assert be.allclose(cached_rays.M, expected_rays.M)
    assert be.allclose(cached_rays.N, expected_rays.N)

    # Check that the cache was populated
    assert len(cached_strategy.cache) == 2

def test_result_correctness(single_thin_lens_optic):
    """Test that the cached strategy returns the same result as the wrapped strategy."""
    paraxial_strategy = ParaxialAimingStrategy()
    cached_strategy = CachedAimingStrategy(paraxial_strategy)
    optic = single_thin_lens_optic
    params = {"optic": optic, "Hx": 0.0, "Hy": 0.1, "Px": 0.0, "Py": 0.1, "wavelength": 0.55}

    expected_rays = paraxial_strategy.aim(**params)
    cached_rays = cached_strategy.aim(**params)

    assert be.allclose(cached_rays.x, expected_rays.x)
    assert be.allclose(cached_rays.y, expected_rays.y)
    assert be.allclose(cached_rays.z, expected_rays.z)
    assert be.allclose(cached_rays.L, expected_rays.L)
    assert be.allclose(cached_rays.M, expected_rays.M)
    assert be.allclose(cached_rays.N, expected_rays.N)
