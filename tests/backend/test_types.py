# tests/backend/test_types.py
"""
Tests for type hint consistency within the Optiland codebase.

This file contains meta-tests that verify that the type hints defined in
different parts of the application (e.g., configuration objects and parameter
TypedDicts) are consistent with each other. This helps to prevent a class
of bugs related to type mismatches during development.
"""
from __future__ import annotations

import importlib
import typing
from dataclasses import fields
from typing import get_type_hints

import pytest

from optiland.surfaces.factories.geometry_factory import GeometryConfig


@pytest.fixture(autouse=True)
def enable_type_checking(monkeypatch):
    """
    A fixture that enables the `typing.TYPE_CHECKING` flag for the duration
    of the test. This is crucial for allowing `get_type_hints` to resolve
    forward references and type hints defined within `if TYPE_CHECKING:` blocks.
    """
    monkeypatch.setattr(typing, "TYPE_CHECKING", True)


def get_object_type_hints(module_name: str, object_name: str) -> dict:
    """
    Dynamically imports a module and retrieves the type hints for a specified
    object within it. The module is reloaded to ensure that the `TYPE_CHECKING`
    flag set by the fixture is effective.

    Args:
        module_name: The name of the module to import.
        object_name: The name of the object (class, function, etc.) within the
                     module to get type hints for.

    Returns:
        A dictionary of type hints for the specified object.
    """
    module = importlib.import_module(module_name)
    importlib.reload(module)
    obj = getattr(module, object_name)
    return get_type_hints(obj)


class TestSurfaceParameters:
    """
    Tests the consistency of type hints between the `GeometryConfig` dataclass
    and the `SurfaceParameters` TypedDict. This ensures that the parameters
    used to create a surface geometry have the same types as the parameters
    defined in the surface's parameter dictionary.
    """

    @pytest.fixture
    def geometry_config_types(self) -> dict:
        """
        Provides the type hints for the `GeometryConfig` dataclass.
        """
        return get_object_type_hints(
            "optiland.surfaces.factories.geometry_factory", "GeometryConfig"
        )

    @pytest.fixture
    def surface_parameters_types(self) -> dict:
        """
        Provides the type hints for the `SurfaceParameters` TypedDict.
        """
        return get_object_type_hints("optiland._types", "SurfaceParameters")

    GEOMETRY_CONFIG_FIELDS = [f.name for f in fields(GeometryConfig)]

    @pytest.mark.parametrize("field_name", GEOMETRY_CONFIG_FIELDS)
    def test_geometry_config_field_in_surface_parameters(
        self, field_name: str, surface_parameters_types: dict
    ):
        """
        Ensures that every field defined in `GeometryConfig` is also present
        in the `SurfaceParameters` TypedDict.
        """
        assert field_name in surface_parameters_types

    @pytest.mark.parametrize("field_name", GEOMETRY_CONFIG_FIELDS)
    def test_equal_geometry_field_types(
        self, field_name: str, geometry_config_types: dict, surface_parameters_types: dict
    ):
        """
        Ensures that for every field present in both `GeometryConfig` and
        `SurfaceParameters`, the type hint is identical.
        """
        field_type = geometry_config_types[field_name]
        parameter_type = surface_parameters_types[field_name]
        assert parameter_type == field_type