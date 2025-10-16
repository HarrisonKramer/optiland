"""Tests for the Surface class.

"""

import pytest
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.surfaces.standard_surface import Surface
from optiland.interactions.phase_model import PhaseInteractionModel
from optiland.phase.grating import GratingPhase
from optiland.rays.real_rays import RealRays
from optiland.geometries.standard import StandardGeometry
from optiland.materials.ideal import IdealMaterial


@pytest.mark.parametrize("backend", be.list_available_backends())
def test_surface_with_phase_interaction(backend):
    """Test that a Surface with a PhaseInteractionModel traces rays
    correctly.

    """
    be.set_backend(backend)
    rays = RealRays(
        x=[0],
        y=[0],
        z=[-1],
        L=[0],
        M=[0],
        N=[1],
        intensity=[1],
        wavelength=[0.5],
    )
    cs = CoordinateSystem()
    geometry = StandardGeometry(coordinate_system=cs, radius=be.inf)
    material = IdealMaterial(n=1.0)
    phase = GratingPhase(period=1.0, order=1)
    interaction_model = PhaseInteractionModel(
        geometry=geometry,
        material_pre=material,
        material_post=material,
        phase_model=phase,
    )
    surface = Surface(
        geometry=geometry,
        material_pre=material,
        material_post=material,
        interaction_model=interaction_model,
    )
    rays = surface.trace(rays)
    assert be.allclose(rays.L, 0.0)
    assert be.allclose(rays.M, 0.5)
    assert be.allclose(rays.N, be.sqrt(1 - 0.5**2))
    assert be.allclose(rays.opd, 1.5)
    assert be.allclose(rays.i, 1.0)
