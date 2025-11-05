"""Unit tests for the refactored, stateless component factories and strategies."""

import pytest
from unittest.mock import MagicMock, ANY

from optiland.coordinate_system import CoordinateSystem
from optiland.materials import IdealMaterial, Material
from optiland.surfaces.factories.coordinate_system_factory import (
    CoordinateSystemFactory,
)
from optiland.surfaces.factories.interaction_model_factory import (
    InteractionModelFactory,
)
from optiland.surfaces.factories.material_factory import MaterialFactory
from optiland.surfaces.factories.strategy_provider import SurfaceStrategyProvider
from optiland.surfaces.factories.strategies.concrete import (
    GratingStrategy,
    ObjectStrategy,
    ParaxialStrategy,
    StandardStrategy,
)
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.standard_surface import Surface


# Test MaterialFactory
def test_material_factory_create_stateless():
    """Verify that the stateless MaterialFactory.create works correctly."""
    assert isinstance(MaterialFactory.create("air"), IdealMaterial)
    schott_bk7 = Material("BK7")
    assert MaterialFactory.create(schott_bk7) is schott_bk7
    assert MaterialFactory.create("mirror") is None
    custom_material = MaterialFactory.create(("BK7", "SCHOTT"))
    assert isinstance(custom_material, Material) and custom_material.name == "BK7"
    with pytest.raises(ValueError):
        MaterialFactory.create(123)


# Test CoordinateSystemFactory
def test_coordinate_system_factory_create_stateless():
    """Verify the stateless CoordinateSystemFactory creates a CS correctly."""
    cs = CoordinateSystemFactory.create(x=1, y=2, z=3, rx=4, ry=5, rz=6)
    assert isinstance(cs, CoordinateSystem)
    assert cs.x == 1 and cs.y == 2 and cs.z == 3
    assert cs.rx == 4 and cs.ry == 5 and cs.rz == 6


# Test InteractionModelFactory
def test_interaction_model_factory_ocp():
    """Verify the OCP-compliant InteractionModelFactory."""
    factory = InteractionModelFactory()
    mock_surface = MagicMock()
    model = factory.create(
        parent_surface=mock_surface,
        interaction_type="refractive_reflective",
        is_reflective=False,
        coating=None,
        bsdf=None,
    )
    assert model is not None


# Test Strategies
def test_standard_strategy():
    strategy = StandardStrategy()
    geom_factory = MagicMock()
    int_factory = MagicMock()
    strategy.create_geometry(geom_factory, MagicMock(), {})
    geom_factory.create.assert_called_with("standard", ANY)
    strategy.create_interaction_model(int_factory, {})
    int_factory.create.assert_called_with(interaction_type="refractive_reflective")
    assert strategy.get_surface_class() is Surface


def test_paraxial_strategy():
    strategy = ParaxialStrategy()
    geom_factory = MagicMock()
    int_factory = MagicMock()
    strategy.create_geometry(geom_factory, MagicMock(), {})
    geom_factory.create.assert_called_with("paraxial", ANY)
    strategy.create_interaction_model(int_factory, {"f": 50.0})
    int_factory.create.assert_called_with(
        interaction_type="thin_lens", focal_length=50.0
    )
    assert strategy.get_surface_class() is Surface


def test_object_strategy():
    strategy = ObjectStrategy()
    geom_factory = MagicMock()
    int_factory = MagicMock()
    strategy.create_geometry(geom_factory, MagicMock(), {})
    geom_factory.create.assert_called_with("plane", ANY)
    assert strategy.create_interaction_model(int_factory, {}) is None
    assert strategy.get_surface_class() is ObjectSurface


# Test StrategyProvider
def test_strategy_provider():
    provider = SurfaceStrategyProvider()
    assert isinstance(provider.get_strategy("standard", 1), StandardStrategy)
    assert isinstance(provider.get_strategy("paraxial", 1), ParaxialStrategy)
    assert isinstance(provider.get_strategy(None, 0), ObjectStrategy)
    assert isinstance(provider.get_strategy("even_asphere", 1), StandardStrategy)
    assert isinstance(provider.get_strategy(None, 1), StandardStrategy)
