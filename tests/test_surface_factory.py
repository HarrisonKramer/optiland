import optiland.backend as be
import pytest
from unittest.mock import MagicMock

from .utils import assert_allclose
from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.materials import IdealMaterial
from optiland.samples.objectives import TessarLens
from optiland.interactions.diffractive_model import DiffractiveInteractionModel
from optiland.surfaces.object_surface import ObjectSurface
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel
from optiland.phase.radial import RadialPhaseProfile
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.factories.surface_factory import SurfaceFactory
from optiland.surfaces.factories.strategy_provider import SurfaceStrategyProvider
from optiland.surfaces.factories.types import SurfaceContext
from optiland.optic import Optic
from optiland.fields import AngleField, Field


class TestSurfaceFactory:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cs_factory = MagicMock()
        self.geom_factory = MagicMock()
        self.mat_factory = MagicMock()
        self.coat_factory = MagicMock()
        self.int_factory = MagicMock()
        self.strategy_provider = SurfaceStrategyProvider()

        self.factory = SurfaceFactory(
            cs_factory=self.cs_factory,
            geom_factory=self.geom_factory,
            mat_factory=self.mat_factory,
            coat_factory=self.coat_factory,
            int_factory=self.int_factory,
            strategy_provider=self.strategy_provider,
        )

        # Mock the return values for simple cases
        self.mat_factory.create.return_value = IdealMaterial(n=1.0)

    def test_create_surface_standard(self, set_test_backend):
        config = {
            "surface_type": "standard",
            "comment": "Standard",
            "is_stop": False,
            "material": "air",
            "thickness": 5,
            "radius": 10,
            "conic": 0,
        }
        context = SurfaceContext(index=1, z=0.0, material_pre=IdealMaterial(n=1.0))
        surface = self.factory.create_surface(config, context)
        assert isinstance(surface, Surface)
        self.geom_factory.create.assert_called()
        self.int_factory.create.assert_called()

    def test_create_surface_object(self, set_test_backend):
        config = {
            "surface_type": "standard",
            "comment": "Object",
            "is_stop": False,
            "material": "air",
            "thickness": 5,
            "radius": 10,
            "conic": 0,
        }
        context = SurfaceContext(index=0, z=-5.0, material_pre=None)
        surface = self.factory.create_surface(config, context)
        assert isinstance(surface, ObjectSurface)

    def test_invalid_surface_type(self, set_test_backend):
        config = {
            "surface_type": "invalid",
            "comment": "Invalid",
            "is_stop": False,
            "material": "air",
            "thickness": 5,
        }
        context = SurfaceContext(index=1, z=0.0, material_pre=IdealMaterial(n=1.0))
        # The error is now raised by the GeometryFactory, not the SurfaceFactory
        self.geom_factory.create.side_effect = ValueError
        with pytest.raises(ValueError):
            self.factory.create_surface(config, context)

    # Note: Many other tests are omitted as they are now integration tests
    # that belong in test_surface_group.py or test_optic.py. The purpose
    # of this file is now to unit test the orchestrator, which we do
    # by asserting that the correct sub-factories are called.
