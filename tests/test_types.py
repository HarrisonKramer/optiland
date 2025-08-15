from dataclasses import fields
from optiland._types import SurfaceParameters
from optiland.surfaces.factories.geometry_factory import GeometryConfig
from typing import get_type_hints
import optiland._types

import pytest


class TestSurfaceParameters:
    GEOMETRY_CONFIG_FIELDS = [f.name for f in fields(GeometryConfig)]
    GEOMETRY_CONFIG_TYPES = get_type_hints(
        GeometryConfig, localns=vars(optiland._types)
    )

    @pytest.mark.parametrize("field_name", GEOMETRY_CONFIG_FIELDS)
    def test_geometry_config_field_in_surface_parameters(self, field_name: str):
        assert field_name in SurfaceParameters.__annotations__

    @pytest.mark.parametrize("field_name", GEOMETRY_CONFIG_FIELDS)
    def test_equal_geometry_field_types(self, field_name: str):
        field_type = self.GEOMETRY_CONFIG_TYPES[field_name]
        parameter_type = SurfaceParameters.__annotations__[field_name]

        assert parameter_type == field_type
