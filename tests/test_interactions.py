"""Tests for the interaction models.

"""

import pytest
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.interactions.phase_model import PhaseInteractionModel
from optiland.phase.grating import GratingPhase
from optiland.rays.real_rays import RealRays
from optiland.geometries.standard import StandardGeometry
from optiland.materials.ideal import IdealMaterial
from .utils import assert_allclose


def test_phase_interaction_model(set_test_backend):
    """Test the PhaseInteractionModel class."""
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
    rays = interaction_model.interact_real_rays(rays)
    assert_allclose(rays.L, 0.0)
    assert_allclose(rays.M, 0.5)
    assert_allclose(rays.N, be.sqrt(1 - 0.5**2))
    assert_allclose(rays.opd, 0.5)
    assert_allclose(rays.i, 1.0)
