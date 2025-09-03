import pytest
import time
import optiland.backend as be
from optiland.optic.optic import Optic
from optiland.rays.aiming.strategy import ModelBasedAimingStrategy, IterativeAimingStrategy
from optiland.physical_apertures import RadialAperture

@pytest.fixture
def single_lens_optic():
    """A simple optic with a single thick lens, suitable for testing aiming."""
    optic = Optic()
    optic.set_field_type("object_height")
    optic.add_field(y=10)
    optic.set_aperture("EPD", 10)
    optic.add_wavelength(value=0.55, is_primary=True)

    optic.add_surface(index=0, thickness=1e10) # Object at infinity
    optic.add_surface(index=1, radius=50, thickness=5, material='N-BK7')
    optic.add_surface(index=2, radius=-50, thickness=95, is_stop=True, aperture=RadialAperture(r_max=5))
    optic.update()
    return optic

def test_model_based_strategy_initialization():
    """Test that the ModelBasedAimingStrategy can be initialized."""
    strategy = ModelBasedAimingStrategy()
    assert strategy is not None
    assert strategy.max_cache_size == 2048
    assert strategy.refit_frequency == 25
    assert strategy.error_tolerance == 1e-6
    assert strategy.model_type == "polynomial"
    assert strategy.min_samples_for_fit == 100

def test_fallback_on_empty_cache(single_lens_optic):
    """Test that the strategy falls back to iterative aiming when the cache is empty."""
    strategy = ModelBasedAimingStrategy()
    optic = single_lens_optic
    Hx, Hy, Px, Py = 0.0, 0.5, 0.0, 0.5
    wavelength = optic.primary_wavelength

    # With an empty cache, this should be equivalent to the iterative strategy
    rays = strategy.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

    # Verify the ray hits the target pupil coordinates accurately
    optic.surface_group.trace(rays)
    stop_idx = optic.surface_group.stop_index
    pupil_x = optic.surface_group.x[stop_idx]
    pupil_y = optic.surface_group.y[stop_idx]
    stop_radius = optic.surface_group.surfaces[stop_idx].aperture.r_max

    assert be.allclose(pupil_x / stop_radius, Px, atol=1e-5)
    assert be.allclose(pupil_y / stop_radius, Py, atol=1e-5)

def test_cache_growth(single_lens_optic):
    """Test that the cache grows with new ray aims."""
    strategy = ModelBasedAimingStrategy(min_samples_for_fit=10)
    optic = single_lens_optic
    wavelength = optic.primary_wavelength

    optic_hash = strategy._get_optic_hash(optic)
    optic_cache = strategy._get_cache_for_optic(optic_hash)

    assert len(optic_cache["cache"]) == 0

    # Aim a few unique rays
    strategy.aim_ray(optic, 0.0, 0.1, 0.0, 0.1, wavelength)
    strategy.aim_ray(optic, 0.0, 0.2, 0.0, 0.2, wavelength)

    assert len(optic_cache["cache"]) == 2

    # Aiming the same ray again should not grow the cache
    strategy.aim_ray(optic, 0.0, 0.1, 0.0, 0.1, wavelength)
    assert len(optic_cache["cache"]) == 2

def test_model_fitting_and_prediction(single_lens_optic):
    """Test that the model is fitted and used for prediction after enough samples."""
    strategy = ModelBasedAimingStrategy(min_samples_for_fit=5, refit_frequency=5)
    optic = single_lens_optic
    wavelength = optic.primary_wavelength

    optic_hash = strategy._get_optic_hash(optic)
    optic_cache = strategy._get_cache_for_optic(optic_hash)

    # Populate the cache to trigger model fitting
    for i in range(5):
        strategy.aim_ray(optic, 0.0, 0.1 * (i + 1), 0.0, 0.1 * (i + 1), wavelength)

    assert optic_cache["model"] is not None

    # Now, aim a ray that is in the cache. It should be predicted correctly.
    # We can't directly check if the model is used, but we can check the result.
    Hx, Hy, Px, Py = 0.0, 0.3, 0.0, 0.3
    predicted_rays = strategy.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

    optic.surface_group.trace(predicted_rays)
    stop_idx = optic.surface_group.stop_index
    pupil_x = optic.surface_group.x[stop_idx]
    pupil_y = optic.surface_group.y[stop_idx]
    stop_radius = optic.surface_group.surfaces[stop_idx].aperture.r_max

    # The prediction should be very close to the target
    assert be.allclose(pupil_x / stop_radius, Px, atol=strategy.error_tolerance)
    assert be.allclose(pupil_y / stop_radius, Py, atol=strategy.error_tolerance)

def test_interpolation_capability(single_lens_optic):
    """Test that the model can generalize to nearby, unseen coordinates."""
    strategy = ModelBasedAimingStrategy(min_samples_for_fit=10, refit_frequency=10, error_tolerance=1e-5)
    optic = single_lens_optic
    wavelength = optic.primary_wavelength

    # Populate cache with a grid of points
    for i in range(4):
        for j in range(4):
            strategy.aim_ray(optic, 0.1*i, 0.1*j, 0.1*i, 0.1*j, wavelength)

    # Aim a ray at an intermediate point not in the cache
    Hx, Hy, Px, Py = 0.15, 0.15, 0.15, 0.15
    interpolated_rays = strategy.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

    optic.surface_group.trace(interpolated_rays)
    stop_idx = optic.surface_group.stop_index
    pupil_x = optic.surface_group.x[stop_idx]
    pupil_y = optic.surface_group.y[stop_idx]
    stop_radius = optic.surface_group.surfaces[stop_idx].aperture.r_max

    # The interpolated prediction should still be reasonably accurate
    assert be.allclose(pupil_x / stop_radius, Px, atol=1e-4)
    assert be.allclose(pupil_y / stop_radius, Py, atol=1e-4)

@pytest.mark.xfail(reason="Performance is environment dependent and this test is flaky")
def test_convergence_speed(single_lens_optic):
    """Test that the model-based strategy is faster after the model is trained."""
    iterative_strategy = IterativeAimingStrategy()
    model_strategy = ModelBasedAimingStrategy(min_samples_for_fit=100, refit_frequency=100)
    optic = single_lens_optic
    wavelength = optic.primary_wavelength

    # Prime the model-based strategy cache
    for i in range(10):
        for j in range(10):
            model_strategy.aim_ray(optic, 0.1*i, 0.1*j, 0.1*i, 0.1*j, wavelength)

    # Time the iterative strategy
    start_time = time.time()
    for _ in range(100):
        iterative_strategy.aim_ray(optic, 0.5, 0.5, 0.5, 0.5, wavelength)
    iterative_time = time.time() - start_time

    # Time the model-based strategy
    start_time = time.time()
    for _ in range(100):
        model_strategy.aim_ray(optic, 0.5, 0.5, 0.5, 0.5, wavelength)
    model_time = time.time() - start_time

    # With a trained model, the model-based strategy should be significantly faster
    assert model_time < iterative_time

def test_cache_eviction(single_lens_optic):
    """Test that the cache size is capped and old entries are evicted."""
    strategy = ModelBasedAimingStrategy(max_cache_size=5, min_samples_for_fit=10, refit_frequency=10)
    optic = single_lens_optic
    wavelength = optic.primary_wavelength

    optic_hash = strategy._get_optic_hash(optic)
    optic_cache = strategy._get_cache_for_optic(optic_hash)

    # Fill the cache
    for i in range(5):
        strategy.aim_ray(optic, 0.1, 0.1 * i, 0.1, 0.1 * i, wavelength)

    assert len(optic_cache["cache"]) == 5
    first_key = next(iter(optic_cache["cache"]))
    assert first_key == (0.1, 0.0, 0.1, 0.0, wavelength)

    # Add one more entry to trigger eviction
    strategy.aim_ray(optic, 0.1, 0.5, 0.1, 0.5, wavelength)

    assert len(optic_cache["cache"]) == 5
    new_first_key = next(iter(optic_cache["cache"]))

    # The original first key should have been evicted
    assert new_first_key != first_key
    assert new_first_key == (0.1, 0.1, 0.1, 0.1, wavelength)
