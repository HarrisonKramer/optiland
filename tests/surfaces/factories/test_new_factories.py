"""Unit tests for the refactored, stateless component factories and strategies."""

import pytest
from unittest.mock import MagicMock

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
    # Test a standard builder
    model = factory.create(
        parent_surface=mock_surface,
        interaction_type="refractive_reflective",
        is_reflective=False,
        coating=None,
        bsdf=None,
    )
    assert model is not None
    # Test with required kwargs
    model_thin_lens = factory.create(
        parent_surface=mock_surface,
        interaction_type="thin_lens",
        is_reflective=False,
        coating=None,
        bsdf=None,
        focal_length=10.0,
    )
    assert model_thin_lens is not None
    # Test for missing kwargs
    with pytest.raises(ValueError):
        factory.create(
            parent_surface=mock_surface,
            interaction_type="thin_lens",
            is_reflective=False,
            coating=None,
            bsdf=None,
        )
    # Test unknown type
    with pytest.raises(ValueError):
        factory.create(
            parent_surface=mock_surface,
            interaction_type="unknown_type",
            is_reflective=False,
            coating=None,
            bsdf=None,
        )


# Test Strategies
@pytest.fixture
def mock_factories():
    """Provides mock factories for strategy tests."""
    return {
        "geom_factory": MagicMock(),
        "int_factory": MagicMock(),
        "cs": MagicMock(),
    }


def test_standard_strategy(mock_factories):
    strategy = StandardStrategy()
    config = {"surface_type": "standard"}
    strategy.create_geometry(mock_factories["geom_factory"], mock_factories["cs"], config)
    mock_factories["geom_factory"].create.assert_called_with(
        "standard", mock_factories["cs"], **config
    )
    strategy.create_interaction_model(mock_factories["int_factory"], config)
    mock_factories["int_factory"].create.assert_called_with(
        interaction_type="refractive_reflective", **config
    )
    assert strategy.get_surface_class() is Surface


def test_paraxial_strategy(mock_factories):
    strategy = ParaxialStrategy()
    config = {}
    strategy.create_geometry(mock_factories["geom_factory"], mock_factories["cs"], config)
    mock_factories["geom_factory"].create.assert_called_with(
        "paraxial", mock_factories["cs"], **config
    )
    strategy.create_interaction_model(mock_factories["int_factory"], config)
    mock_factories["int_factory"].create.assert_called_with(
        interaction_type="thin_lens", **config
    )
    assert strategy.get_surface_class() is Surface


def test_grating_strategy(mock_factories):
    strategy = GratingStrategy()
    config = {}
    strategy.create_geometry(mock_factories["geom_factory"], mock_factories["cs"], config)
    mock_factories["geom_factory"].create.assert_called_with(
        "grating", mock_factories["cs"], **config
    )
    strategy.create_interaction_model(mock_factories["int_factory"], config)
    mock_factories["int_factory"].create.assert_called_with(
        interaction_type="diffractive", **config
    )
    assert strategy.get_surface_class() is Surface


def test_object_strategy(mock_factories):
    strategy = ObjectStrategy()
    config = {}
    strategy.create_geometry(mock_factories["geom_factory"], mock_factories["cs"], config)
    mock_factories["geom_factory"].create.assert_called_with(
        "plane", mock_factories["cs"], **config
    )
    assert strategy.create_interaction_model(mock_factories["int_factory"], config) is None
    assert strategy.get_surface_class() is ObjectSurface


# Test StrategyProvider
def test_strategy_provider():
    provider = SurfaceStrategyProvider()
    assert isinstance(provider.get_strategy("standard"), StandardStrategy)
    assert isinstance(provider.get_strategy("paraxial"), ParaxialStrategy)
    assert isinstance(provider.get_strategy("object"), ObjectStrategy)
    assert isinstance(provider.get_strategy("grating"), GratingStrategy)
    # Test fallback for geometric types
    assert isinstance(provider.get_strategy("even_asphere"), StandardStrategy)
    # Test fallback for None
    assert isinstance(provider.get_strategy(None), StandardStrategy)
