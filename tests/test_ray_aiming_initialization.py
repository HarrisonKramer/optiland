
import numpy as np
import pytest

from optiland.optic import Optic
from optiland.rays.ray_aiming.initialization import (
    FloatByStopStrategy,
    ParaxialReferenceStrategy,
    RealReferenceStrategy,
    get_stop_radius_strategy,
)


def test_float_by_stop_strategy(set_test_backend):
    optic = Optic()
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(index=1, radius=np.inf, thickness=10, is_stop=True)
    optic.add_surface(index=2)

    # Set explicit aperture on stop
    from optiland.physical_apertures.radial import RadialAperture
    optic.surface_group.surfaces[1].aperture = RadialAperture(2.5)
    optic.set_aperture('float_by_stop_size', 1.0)
    
    strategy = FloatByStopStrategy(optic)
    r = strategy.calculate_stop_radius()
    assert r == 2.5

    # Factory check
    s = get_stop_radius_strategy(optic, 'paraxial')
    assert isinstance(s, FloatByStopStrategy)


def test_paraxial_reference_strategy(set_test_backend):
    # Create an aberrated lens where paraxial approximation differs
    optic = Optic()
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(index=1, radius=-10.0, thickness=5.0, material='N-BK7')
    optic.add_surface(index=2, radius=np.inf, thickness=20.0)
    # Stop
    optic.add_surface(index=3, radius=np.inf, thickness=10.0, is_stop=True)
    optic.add_surface(index=4)

    optic.set_aperture('EPD', 10.0)
    optic.add_wavelength(0.55)

    strategy = ParaxialReferenceStrategy(optic)
    r_par = strategy.calculate_stop_radius()
    # Rays diverge after S1, so stop radius > EPD/2 (5.0)
    assert r_par > 5.0


def test_real_reference_strategy_vs_paraxial_simple(set_test_backend):
    # Use a single spherical surface to demonstrate Spherical Aberration
    # causing Real != Paraxial
    optic = Optic()
    # Pfx lens: R=20, n=1.5
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(index=1, radius=20.0, thickness=20.0, material='N-BK7')
    optic.add_surface(index=2, radius=np.inf, thickness=10.0, is_stop=True)

    optic.set_aperture('EPD', 36.0)  # High NA to ensure aberration
    optic.add_wavelength(1.0)

    strat_real = RealReferenceStrategy(optic)
    r_real = strat_real.calculate_stop_radius()

    strat_par = ParaxialReferenceStrategy(optic)
    r_par = strat_par.calculate_stop_radius()

    # With large aperture, spherical aberration is significant.
    # We expect some difference. If fallback happened, diff is exactly 0.
    assert r_real != r_par

    # Factory check
    s = get_stop_radius_strategy(optic, 'robust')
    assert isinstance(s, RealReferenceStrategy)


def test_finite_object_real_strategy(set_test_backend):
    optic = Optic()
    # Finite Object distance 20
    optic.add_surface(index=0, radius=np.inf, thickness=20.0)
    optic.add_surface(index=1, radius=20.0, thickness=5.0, material='N-BK7')
    optic.add_surface(index=2, radius=np.inf, thickness=10.0, is_stop=True)
    optic.add_surface(index=3)

    optic.set_aperture('EPD', 5.0)
    optic.add_wavelength(0.55)

    strat_real = RealReferenceStrategy(optic)
    r_real = strat_real.calculate_stop_radius()

    assert r_real > 0.0
