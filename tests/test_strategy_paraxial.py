import pytest
import optiland.backend as be
from optiland.optic.optic import Optic
from optiland.rays.aiming.strategy import ParaxialAimingStrategy
from optiland.samples import objectives

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

def test_paraxial_aiming_strategy_compare_to_legacy(single_thin_lens_optic):
    """Test ParaxialAimingStrategy by comparing to legacy paraxial trace."""
    strategy = ParaxialAimingStrategy()
    optic = single_thin_lens_optic

    Hy, Py = 0.1, 0.1  # small values for better paraxial approximation
    wavelength = optic.primary_wavelength

    # 1. Trace with new strategy (real rays)
    initial_rays = strategy.aim_ray(optic, Hx=0, Hy=Hy, Px=0, Py=Py, wavelength=wavelength)
    optic.surface_group.trace(initial_rays)
    y_new = be.copy(optic.surface_group.y)

    # 2. Trace with legacy method (paraxial rays)
    optic.paraxial.trace(Hy, Py, wavelength)
    y_legacy = optic.surface_group.y

    assert be.allclose(y_new, y_legacy, atol=1e-5)

def test_paraxial_aiming_strategy_vectorized(single_thin_lens_optic):
    """Test ParaxialAimingStrategy with vectorized input."""
    strategy = ParaxialAimingStrategy()
    optic = single_thin_lens_optic
    optic.update()

    Hx = be.array([0.0, 0.1])
    Hy = be.array([0.0, 0.1])
    Px = be.array([0.0, 0.1])
    Py = be.array([0.1, 0.1])
    wavelength = optic.primary_wavelength

    rays = strategy.aim_ray(optic, Hx, Hy, Px, Py, wavelength)

    assert rays.x.shape == (2,)
    assert not be.any(be.isnan(rays.x))
    assert not be.any(be.isnan(rays.y))
    assert not be.any(be.isnan(rays.z))
    assert not be.any(be.isnan(rays.L))
    assert not be.any(be.isnan(rays.M))
    assert not be.any(be.isnan(rays.N))

    # check first ray against scalar calculation
    ray0_scalar = strategy.aim_ray(optic, Hx[0], Hy[0], Px[0], Py[0], wavelength)
    assert be.allclose(rays.x[0], ray0_scalar.x)
    assert be.allclose(rays.y[0], ray0_scalar.y)
    assert be.allclose(rays.L[0], ray0_scalar.L)
    assert be.allclose(rays.M[0], ray0_scalar.M)
