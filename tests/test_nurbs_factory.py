import pytest
from unittest.mock import MagicMock
import optiland.backend as be

from optiland.surfaces.factories.surface_factory import SurfaceFactory
from optiland.surfaces.factories.strategy_provider import SurfaceStrategyProvider
from optiland.surfaces.factories.types import SurfaceContext
from optiland.surfaces import Surface


class MockSurfaceGroup:
    def __init__(self, num_surfaces):
        self.num_surfaces = num_surfaces


@pytest.mark.parametrize("backend", be.list_available_backends())
def test_nurbs_factory(set_test_backend, backend):
    be.set_backend(backend)
    config = {
        "surface_type": "nurbs",
        "radius": 100,
        "conic": -1,
        "comment": "NURBS test surface",
        "is_stop": False,
        "material": "air",
    }
    context = SurfaceContext(index=1, z=0.0, material_pre=None)

    # We need to mock the sub-factories
    cs_factory = MagicMock()
    geom_factory = MagicMock()
    mat_factory = MagicMock()
    coat_factory = MagicMock()
    int_factory = MagicMock()
    strategy_provider = SurfaceStrategyProvider()  # Use the real one

    factory = SurfaceFactory(
        cs_factory=cs_factory,
        geom_factory=geom_factory,
        mat_factory=mat_factory,
        coat_factory=coat_factory,
        int_factory=int_factory,
        strategy_provider=strategy_provider,
    )

    surface = factory.create_surface(config, context)

    # We can only assert that the correct calls were made
    cs_factory.create.assert_called_once()
    geom_factory.create.assert_called_once()
    mat_factory.create.assert_called_once()
    int_factory.create.assert_called_once()
