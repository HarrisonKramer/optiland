# tests/geometries/test_base.py
"""
Tests for the base geometry functionality in optiland.geometries.
"""
import pytest

from optiland import geometries


def test_unknown_geometry(set_test_backend):
    """
    Tests that attempting to create a geometry from a dictionary with an
    unknown 'type' raises a ValueError.
    """
    with pytest.raises(ValueError):
        geometries.BaseGeometry.from_dict({"type": "UnknownGeometry"})