"""Tests for the phase module.

"""

import pytest
import optiland.backend as be
from optiland.phase.grating import GratingPhase
from optiland.phase.radial import RadialPhase
from optiland.rays.real_rays import RealRays


@pytest.mark.parametrize("backend", be.list_available_backends())
def test_grating_phase(backend):
    """Test the GratingPhase class."""
    be.set_backend(backend)
    rays = RealRays(
        x=[0],
        y=[0],
        z=[0],
        L=[0],
        M=[0],
        N=[1],
        intensity=[1],
        wavelength=[0.5],
    )
    phase = GratingPhase(period=1.0, order=1)
    L, M, N, opd = phase.phase_calc(rays, 0, 0, -1, 1, 1)
    assert be.allclose(L, 0.0)
    assert be.allclose(M, 0.5)
    assert be.allclose(N, be.sqrt(1 - 0.5**2))
    assert be.allclose(opd, 0.5)


@pytest.mark.parametrize("backend", be.list_available_backends())
def test_radial_phase(backend):
    """Test the RadialPhase class."""
    be.set_backend(backend)
    rays = RealRays(
        x=[1],
        y=[0],
        z=[0],
        L=[0],
        M=[0],
        N=[1],
        intensity=[1],
        wavelength=[0.5],
    )
    phase = RadialPhase(order=1, coefficients=[0.1])
    L, M, N, opd = phase.phase_calc(rays, 0, 0, -1, 1, 1)
    assert be.allclose(L, 0.1, atol=1e-6)
    assert be.allclose(M, 0.0, atol=1e-6)
    assert be.allclose(N, be.sqrt(1 - 0.1**2), atol=1e-6)
    assert be.allclose(opd, 0.1)
