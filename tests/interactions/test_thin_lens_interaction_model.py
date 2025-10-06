# tests/interactions/test_thin_lens_interaction_model.py
"""
Tests for the ThinLensInteractionModel class in optiland.interactions.
"""
import pytest

import optiland.backend as be
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel
from optiland.rays import ParaxialRays, RealRays
from ..utils import assert_allclose


@pytest.fixture
def thin_lens_model():
    """
    Provides a ThinLensInteractionModel instance with a focal length of 100
    for testing.
    """
    return ThinLensInteractionModel(f=100.0)


class TestThinLensInteractionModel:
    """
    Tests the ThinLensInteractionModel, which simulates the effect of a thin
    lens on rays.
    """

    def test_interact_paraxial(self, set_test_backend, thin_lens_model):
        """
        Tests the interaction with paraxial rays. The model should change the
        ray angle based on the lens's power.
        """
        # A paraxial ray at height y=10, parallel to the axis (u=0)
        rays = ParaxialRays(y=be.array([10.0]), u=be.array([0.0]))
        interacted_rays = thin_lens_model.interact(rays)
        # Expected new angle u' = u - y/f = 0 - 10/100 = -0.1
        assert_allclose(interacted_rays.u, -0.1)
        # Height should be unchanged by the interaction
        assert_allclose(interacted_rays.y, 10.0)

    def test_interact_real(self, set_test_backend, thin_lens_model):
        """
        Tests the interaction with real rays. The model should change the ray
        direction cosine M based on the lens's power.
        """
        # A real ray at height y=10, parallel to the axis (M=0)
        rays = RealRays(x=0, y=10.0, z=0, L=0, M=0, N=1)
        interacted_rays = thin_lens_model.interact(rays)
        # Expected new direction cosine M' = M - y/f = 0 - 10/100 = -0.1
        assert_allclose(interacted_rays.M, -0.1)
        # Other properties should be unchanged by this simplified model
        assert_allclose(interacted_rays.y, 10.0)
        assert_allclose(interacted_rays.L, 0.0)

    def test_to_dict(self, set_test_backend, thin_lens_model):
        """
        Tests the serialization of a ThinLensInteractionModel instance to a
        dictionary.
        """
        d = thin_lens_model.to_dict()
        assert d["type"] == "ThinLensInteractionModel"
        assert d["f"] == 100.0

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of a ThinLensInteractionModel instance from
        a dictionary.
        """
        d = {"type": "ThinLensInteractionModel", "f": 100.0}
        model = ThinLensInteractionModel.from_dict(d)
        assert isinstance(model, ThinLensInteractionModel)
        assert model.f == 100.0