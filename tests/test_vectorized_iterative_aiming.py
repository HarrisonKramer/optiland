import pytest
import optiland.backend as be
from optiland.optic.optic import Optic
from optiland.aiming.strategies import IterativeAimingStrategy
from optiland.samples.objectives import CookeTriplet

@pytest.fixture
def cooke_triplet_optic():
    """A Cooke triplet optic."""
    return CookeTriplet()

def test_vectorized_iterative_aiming_with_cooke_triplet(cooke_triplet_optic):
    """Test that vectorized iterative aiming converges for a Cooke triplet."""
    strategy = IterativeAimingStrategy()
    optic = cooke_triplet_optic
    wavelength = optic.primary_wavelength

    # Use a few different fields
    fields_xy = be.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    num_fields = fields_xy.shape[0]

    # Generate random points in the pupil
    num_pupil_points = 10
    be.set_backend('numpy') # use numpy for random number generation
    rng = be.random.default_rng(12345)
    pupil_xy = rng.uniform(-1.0, 1.0, (num_pupil_points, 2))

    # Create all combinations of fields and pupil points
    Hx = be.repeat(fields_xy[:, 0], num_pupil_points)
    Hy = be.repeat(fields_xy[:, 1], num_pupil_points)
    Px = be.tile(pupil_xy[:, 0], num_fields)
    Py = be.tile(pupil_xy[:, 1], num_fields)


    rays = strategy.aim(optic, Hx, Hy, Px, Py, wavelength)
    optic.surface_group.trace(rays)

    stop_idx = optic.surface_group.stop_index
    pupil_x = optic.surface_group.x[stop_idx]
    pupil_y = optic.surface_group.y[stop_idx]

    stop_radius = strategy._stop_radius(optic)
    norm_pupil_x = pupil_x / stop_radius
    norm_pupil_y = pupil_y / stop_radius

    assert be.allclose(norm_pupil_x, Px, atol=1e-6)
    assert be.allclose(norm_pupil_y, Py, atol=1e-6)
