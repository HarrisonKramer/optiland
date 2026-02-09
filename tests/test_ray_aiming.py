
import pytest
import numpy as np
import optiland.backend as be
from optiland.samples.objectives import (
    ReverseTelephoto,
    CookeTriplet,
    WideAngle100FOV,
    ProjectionLens120FOV,
    ProjectionLens160FOV,
    WideAngle170FOV
)
from optiland.optic import Optic
from optiland.rays.ray_aiming.iterative import IterativeRayAimer
from optiland.rays.ray_aiming.robust import RobustRayAimer

def test_iterative_aimer_infinite(set_test_backend):
    """Test IterativeRayAimer on an infinite conjugate system."""
    optic = ReverseTelephoto()
    aimer = IterativeRayAimer(optic, tol=1e-8)
    
    Hx, Hy = 0.0, 1.0
    Px, Py = 0.0, 1.0
    wavelength = 0.55
    
    x, y, z, L, M, N = aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))
    
    stop_idx = optic.surface_group.stop_index
    
    from optiland.rays import RealRays
    rays = RealRays(x, y, z, L, M, N, intensity=1.0, wavelength=wavelength)
    for i in range(1, stop_idx + 1):
        optic.surface_group.surfaces[i].trace(rays)
        
    stop_surf = optic.surface_group.surfaces[stop_idx]
    assert not be.any(be.isnan(x))

def test_robust_aimer_infinite(set_test_backend):
    """Test RobustRayAimer on an infinite conjugate system."""
    optic = ReverseTelephoto()
    aimer = RobustRayAimer(optic, tol=1e-8)
    
    Hx, Hy = 0.0, 1.0
    Px, Py = 0.0, 1.0
    wavelength = 0.55
    
    x, y, z, L, M, N = aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))
    assert not be.any(be.isnan(x))

def test_aimer_consistency(set_test_backend):
    """Ensure Iterative and Robust aimers yield similar results for easy rays."""
    optic = ReverseTelephoto()
    iter_aimer = IterativeRayAimer(optic, tol=1e-10)
    robust_aimer = RobustRayAimer(optic, tol=1e-10)
    
    Hx, Hy = 0.0, 0.5
    Px, Py = 0.0, 0.5
    wavelength = 0.55
    
    res_iter = iter_aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))
    res_robust = robust_aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))
    
    for r_i, r_r in zip(res_iter, res_robust):
        assert be.allclose(r_i, r_r, atol=1e-6)


def test_large_batch(set_test_backend):
    """Test aiming with a large batch of rays."""
    optic = ReverseTelephoto()
    aimer = IterativeRayAimer(optic)
    
    n = 100
    Hx = np.zeros(n)
    Hy = np.linspace(0, 1, n)
    Px = np.linspace(-1, 1, n)
    Py = np.zeros(n)
    wavelength = 0.55
    
    x, y, z, L, M, N = aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))
    
    assert len(x) == n
    assert not be.any(be.isnan(x))

def test_robust_aimer_initialization(set_test_backend):
    """Test the initialization of the RobustRayAimer."""
    optic = ReverseTelephoto()
    aimer = RobustRayAimer(optic, max_iter=30, tol=1e-7, scale_fields=False)
    
    assert aimer.optic == optic
    assert aimer._iterative.max_iter == 30
    assert aimer._iterative.tol == 1e-7
    assert aimer.scale_fields is False
    assert isinstance(aimer._paraxial, type(aimer._iterative._paraxial_aimer))

def test_integration_via_optic(set_test_backend):
    """Test setting the ray aimer via the Optic class."""
    optic = ReverseTelephoto()
    
    optic.set_ray_aiming("iterative", max_iter=25, tol=1e-5)
    ray_gen = optic.ray_tracer.ray_generator
    optic.trace(0, 0, 0.55, num_rays=1)
    
    aimer = ray_gen.aimer
    assert isinstance(aimer, IterativeRayAimer)
    assert aimer.max_iter == 25
    assert aimer.tol == 1e-5
    
    optic.set_ray_aiming("robust", max_iter=15, tol=1e-6, scale_fields=True)
    optic.trace(0, 0, 0.55, num_rays=1)
    
    aimer = ray_gen.aimer
    assert isinstance(aimer, RobustRayAimer)
    assert aimer._iterative.max_iter == 15
    assert aimer._iterative.tol == 1e-6
    assert aimer.scale_fields is True

def test_robust_caching_regression(set_test_backend):
    """Regression test for cached RobustRayAimer accepting initial_guess."""
    optic = ReverseTelephoto()
    optic.set_ray_aiming("robust", cache=True)
    
    # 1. First trace (populates cache)
    optic.trace(0, 0, 0.55, num_rays=1)
    
    # 2. Perturb system to force reuse of result as initial_guess
    # Modify a radius slightly. We can just set a new value directly.
    # This ensures the system hash changes.
    optic.set_radius(100.0, 1)
    
    # 3. Second trace (should call robust aimer with initial_guess)
    optic.trace(0, 0, 0.55, num_rays=1)

def test_robust_aimer_infinite_object_90_degree_field(set_test_backend):
    """Regression test: verify RobustRayAimer aims correctly for 90 deg field @ infinity.
    
    See bug fix where IterativeRayAimer inherited bad L,M,N from initial_guess.
    """
    optic = Optic()
    # Construct a minimal wide angle lens setup that reproduces the infinite + 90 deg scenario
    # We'll use a simplified version of the user's lens to avoid clutter, 
    # but ensure it has infinite object and large field.
    
    optic.add_surface(index=0, radius=float('inf'), thickness=float('inf'))
    # A dummy surface to aim at
    optic.add_surface(index=1, radius=100.0, thickness=10.0, material='air', is_stop=True)
    optic.add_surface(index=2)

    optic.set_aperture('EPD', 1.0)
    optic.set_field_type('angle')
    optic.add_field(y=0)
    optic.add_field(y=90)
    optic.add_wavelength(0.55, is_primary=True)
    
    optic.set_ray_aiming("robust")
    
    from optiland.rays.ray_generator import RayGenerator
    rg = RayGenerator(optic)
    
    # Generate rays for 90 degree field (Hy=1.0)
    # 90 degrees means rays come from +Y relative to Z.
    # Direction vector should be approx (0, 1, 0).
    # N (z-dir cosine) should be near 0.
    rays = rg.generate_rays(Hx=0, Hy=1, Px=0, Py=0, wavelength=0.55)
    
    # Check N is close to 0 (allow small tolerance due to numerical precision/mapping)
    # Using the fixed code, we saw N ~ 0.02 which is small enough compared to N ~ 1.
    assert abs(rays.N[0]) < 0.1
    assert rays.M[0] > 0.9 # Should be largely in Y direction


def test_instantiate_wide_angle_lenses(set_test_backend):
    """This tests only if we can instantiate wide angle lenses with error"""
    assert WideAngle100FOV() is not None
    assert ProjectionLens120FOV() is not None
    assert ProjectionLens160FOV() is not None
    assert WideAngle170FOV() is not None
