from __future__ import annotations

import importlib
import typing
from dataclasses import fields
from typing import get_type_hints

import pytest

from optiland.surfaces.factories.geometry_factory import GeometryConfig


@pytest.fixture(autouse=True)
def enable_type_checking(monkeypatch):
    monkeypatch.setattr(typing, "TYPE_CHECKING", True)


def get_object_type_hints(module_name: str, object_name: str):
    """Retrieve type hints for a specific object from a module.

    The module is imported and reloaded to ensure TYPE_CHECKING is enabled.
    """
    module = importlib.import_module(module_name)
    importlib.reload(
        module
    )  # Ensure the module is reloaded to get the latest definitions
    obj = getattr(module, object_name)

    return get_type_hints(obj)


class TestSurfaceParameters:
    @pytest.fixture
    def geometry_config_types(self):
        return get_object_type_hints(
            "optiland.surfaces.factories.geometry_factory", "GeometryConfig"
        )

    @pytest.fixture
    def surface_parameters_types(self):
        return get_object_type_hints("optiland._types", "SurfaceParameters")

    GEOMETRY_CONFIG_FIELDS = [f.name for f in fields(GeometryConfig)]

    @pytest.mark.parametrize("field_name", GEOMETRY_CONFIG_FIELDS)
    def test_geometry_config_field_in_surface_parameters(
        self, field_name: str, surface_parameters_types
    ):
        assert field_name in surface_parameters_types

    @pytest.mark.parametrize("field_name", GEOMETRY_CONFIG_FIELDS)
    def test_equal_geometry_field_types(
        self, field_name: str, geometry_config_types, surface_parameters_types
    ):
        field_type = geometry_config_types[field_name]
        parameter_type = surface_parameters_types[field_name]

        assert parameter_type == field_type
