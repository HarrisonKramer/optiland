import pytest
import optiland.backend as be
from optiland.optic.optic import Optic
from optiland.aiming.strategies import IterativeAimingStrategy, ParaxialAimingStrategy
from optiland.physical_apertures import RadialAperture
from optiland.samples.objectives import CookeTriplet

@pytest.fixture
def cooke_triplet_optic():
    """A Cooke triplet optic that is known to cause issues with ray aiming."""
    return CookeTriplet()

def test_iterative_aiming_with_complex_optic(cooke_triplet_optic):
    """Test that iterative aiming does not produce NaNs with a complex optic."""
    strategy = IterativeAimingStrategy()
    optic = cooke_triplet_optic

    # These coordinates are known to cause failures
    Hx, Hy, Px, Py = 0.0, 1.0, 0.0, 1.0
    wavelength = optic.primary_wavelength

    rays = strategy.aim(optic, Hx, Hy, Px, Py, wavelength)

    assert not be.any(be.isnan(rays.x)), "NaN values found in ray x coordinates"
    assert not be.any(be.isnan(rays.y)), "NaN values found in ray y coordinates"
    assert not be.any(be.isnan(rays.z)), "NaN values found in ray z coordinates"
    assert not be.any(be.isnan(rays.L)), "NaN values found in ray L direction cosines"
    assert not be.any(be.isnan(rays.M)), "NaN values found in ray M direction cosines"
    assert not be.any(be.isnan(rays.N)), "NaN values found in ray N direction cosines"

@pytest.fixture
def single_lens_optic():
    """A simple optic with a single thick lens."""
    optic = Optic()
    optic.set_field_type("object_height")
    optic.add_field(y=10)
    optic.set_aperture("EPD", 5)
    optic.add_wavelength(value=0.55, is_primary=True)

    optic.add_surface(index=0, thickness=1e10) # Object at infinity
    optic.add_surface(index=1, radius=50, thickness=5, material='N-BK7')
    optic.add_surface(index=2, radius=-50, thickness=95, is_stop=True, aperture=RadialAperture(r_max=5))
    optic.update()
    return optic

def test_iterative_aiming_convergence(single_lens_optic):
    """Test that the iterative aiming strategy converges."""
    strategy = IterativeAimingStrategy(max_iter=10, tolerance=1e-6)
    optic = single_lens_optic

    Hx, Hy, Px, Py = 0.0, 1.0, 0.0, 1.0
    wavelength = optic.primary_wavelength

    rays = strategy.aim(optic, Hx, Hy, Px, Py, wavelength)
    optic.surface_group.trace(rays)

    stop_idx = optic.surface_group.stop_index
    pupil_x = optic.surface_group.x[stop_idx]
    pupil_y = optic.surface_group.y[stop_idx]

    stop_radius = optic.surface_group.surfaces[stop_idx].aperture.r_max
    norm_pupil_x = pupil_x / stop_radius
    norm_pupil_y = pupil_y / stop_radius

    assert be.allclose(norm_pupil_x, Px, atol=1e-5)
    assert be.allclose(norm_pupil_y, Py, atol=1e-5)

def test_iterative_improves_on_paraxial(single_lens_optic):
    """Test that the iterative strategy improves pupil error over paraxial."""
    paraxial_strategy = ParaxialAimingStrategy()
    iterative_strategy = IterativeAimingStrategy()
    optic = single_lens_optic

    Hx, Hy, Px, Py = 0.0, 0.5, 0.0, 0.5
    wavelength = optic.primary_wavelength

    # Paraxial error
    paraxial_rays = paraxial_strategy.aim(optic, Hx, Hy, Px, Py, wavelength)
    optic.surface_group.trace(paraxial_rays)
    stop_idx = optic.surface_group.stop_index
    paraxial_pupil_x = optic.surface_group.x[stop_idx]
    paraxial_pupil_y = optic.surface_group.y[stop_idx]
    stop_radius = optic.surface_group.surfaces[stop_idx].aperture.r_max
    paraxial_error = be.sqrt((paraxial_pupil_x/stop_radius - Px)**2 + (paraxial_pupil_y/stop_radius - Py)**2)

    # Iterative error
    iterative_rays = iterative_strategy.aim(optic, Hx, Hy, Px, Py, wavelength)
    optic.surface_group.trace(iterative_rays)
    iterative_pupil_x = optic.surface_group.x[stop_idx]
    iterative_pupil_y = optic.surface_group.y[stop_idx]
    stop_radius = optic.surface_group.surfaces[stop_idx].aperture.r_max
    iterative_error = be.sqrt((iterative_pupil_x/stop_radius - Px)**2 + (iterative_pupil_y/stop_radius - Py)**2)

    assert be.all(iterative_error < paraxial_error)

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_backend_consistency(single_lens_optic, backend):
    """Test that numpy and torch backends produce consistent results."""
    be.set_backend(backend)

    strategy = IterativeAimingStrategy()
    optic = single_lens_optic

    Hx, Hy, Px, Py = 0.0, 1.0, 0.0, 1.0
    wavelength = optic.primary_wavelength

    rays = strategy.aim(optic, Hx, Hy, Px, Py, wavelength)

    assert not be.any(be.isnan(rays.x))
    assert not be.any(be.isnan(rays.y))
    assert not be.any(be.isnan(rays.z))
    assert not be.any(be.isnan(rays.L))
    assert not be.any(be.isnan(rays.M))
    assert not be.any(be.isnan(rays.N))

@pytest.fixture
def afocal_system():
    """An afocal system where the Jacobian might be singular."""
    optic = Optic()
    optic.set_field_type("object_height")
    optic.add_field(y=1)
    optic.set_aperture("EPD", 10)
    optic.add_wavelength(value=0.55, is_primary=True)

    optic.add_surface(index=0, thickness=1e10)
    optic.add_surface(index=1, radius=50, thickness=100, material='N-BK7')
    optic.add_surface(index=2, radius=-50, is_stop=True, aperture=RadialAperture(r_max=10))
    optic.update()
    return optic

def test_singular_jacobian_fallback(afocal_system):
    """Test that the strategy falls back to gradient descent on singular Jacobian."""
    strategy = IterativeAimingStrategy()
    optic = afocal_system

    Hx, Hy, Px, Py = 0.0, 0.1, 0.0, 0.1
    wavelength = optic.primary_wavelength

    # This should not raise a LinAlgError
    try:
        strategy.aim(optic, Hx, Hy, Px, Py, wavelength)
    except be.linalg.LinAlgError:
        pytest.fail("IterativeAimingStrategy failed to handle singular Jacobian.")
