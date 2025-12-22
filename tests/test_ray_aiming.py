
import pytest
import numpy as np
import optiland.backend as be
from optiland.samples.objectives import ReverseTelephoto, CookeTriplet
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
    assert not np.any(np.isnan(x))

def test_robust_aimer_infinite(set_test_backend):
    """Test RobustRayAimer on an infinite conjugate system."""
    optic = ReverseTelephoto()
    aimer = RobustRayAimer(optic, tol=1e-8)
    
    Hx, Hy = 0.0, 1.0
    Px, Py = 0.0, 1.0
    wavelength = 0.55
    
    x, y, z, L, M, N = aimer.aim_rays((Hx, Hy), wavelength, (Px, Py))
    assert not np.any(np.isnan(x))

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
    assert not np.any(np.isnan(x))
