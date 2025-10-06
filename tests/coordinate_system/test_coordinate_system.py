# tests/coordinate_system/test_coordinate_system.py
"""
Tests for the CoordinateSystem class in optiland.coordinate_system.

This file verifies the functionality of the CoordinateSystem class, including
initialization, coordinate transformations (localize and globalize), and
serialization.
"""
import optiland.backend as be
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from ..utils import assert_allclose


def test_coordinate_system_init(set_test_backend):
    """
    Tests the initialization of a CoordinateSystem with both default and
    custom values, including a reference coordinate system.
    """
    # Test default initialization
    cs_default = CoordinateSystem()
    assert cs_default.x == 0
    assert cs_default.reference_cs is None

    # Test initialization with custom values and a reference
    ref_cs = CoordinateSystem(1, 2, 3)
    cs_custom = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, ref_cs)
    assert cs_custom.x == 10
    assert cs_custom.rx == 0.5
    assert cs_custom.reference_cs == ref_cs


def test_coordinate_system_localize(set_test_backend):
    """
    Tests the `localize` method, which transforms ray coordinates from a
    parent (global or reference) system to the local system.
    """
    # Test localization without a reference CS (from global)
    cs1 = CoordinateSystem(1, -1.0, 2.0)
    rays1 = RealRays(x=1, y=2, z=5, L=0, M=0, N=1)
    cs1.localize(rays1)
    assert rays1.x == 0.0
    assert rays1.y == 3.0
    assert rays1.z == 3.0

    # Test localization with a reference CS
    ref_cs = CoordinateSystem(5, 5, 5, 0.2, 0.3, 0.4)
    cs2 = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, ref_cs)
    rays2 = RealRays(x=1, y=2, z=3, L=0.1, M=0.2, N=0.3)
    cs2.localize(rays2)
    assert_allclose(rays2.x, -1.826215)
    assert_allclose(rays2.y, -26.51361)
    assert_allclose(rays2.z, -32.55361)
    assert_allclose(rays2.L, -0.0122750)
    assert_allclose(rays2.M, 0.2697627)
    assert_allclose(rays2.N, 0.2589931)


def test_coordinate_system_globalize(set_test_backend):
    """
    Tests the `globalize` method, which transforms ray coordinates from the
    local system to the parent (global or reference) system.
    """
    # Test globalization without a reference CS (to global)
    cs1 = CoordinateSystem(1, -1.0, 2.0)
    rays1 = RealRays(x=0.0, y=3.0, z=3.0, L=0.0, M=0.0, N=1.0)
    cs1.globalize(rays1)
    assert_allclose(rays1.x, 1.0)
    assert_allclose(rays1.y, 2.0)
    assert_allclose(rays1.z, 5.0)

    # Test globalization with a reference CS
    ref_cs = CoordinateSystem(5, 5, 5, 0.2, 0.3, 0.4)
    cs2 = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, ref_cs)
    rays2 = RealRays(x=-23.636, y=-25.402, z=-23.083, L=0.231, M=0.237, N=0.174)
    cs2.globalize(rays2)
    assert_allclose(rays2.x, 4.654, atol=1e-3)
    assert_allclose(rays2.y, -12.502, atol=1e-3)
    assert_allclose(rays2.z, 21.517, atol=1e-3)
    assert_allclose(rays2.L, 0.086, atol=1e-3)
    assert_allclose(rays2.M, 0.352, atol=1e-3)
    assert_allclose(rays2.N, 0.089, atol=1e-3)


def test_coordinate_system_transform(set_test_backend):
    """
    Tests the `get_effective_transform` method, which computes the total
    translation and rotation from the global system to the local system,
    accounting for any reference systems.
    """
    cs1 = CoordinateSystem(1, -1.0, 2.0)
    cs2 = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, cs1)
    eff_translation, eff_rot_mat = cs2.get_effective_transform()
    assert be.allclose(eff_translation, be.array([11, 19, 32]))
    rot_mat = be.array(
        [[0.6312, -0.3583, 0.6878],
         [0.5316, 0.8456, -0.0474],
         [-0.5646, 0.3956, 0.7243]]
    )
    assert be.allclose(eff_rot_mat, rot_mat, atol=1e-4)


def test_coordinate_system_to_dict(set_test_backend):
    """
    Tests the serialization of a CoordinateSystem instance to a dictionary.
    """
    cs = CoordinateSystem(1, -1.0, 2.0)
    cs_dict = cs.to_dict()
    assert cs_dict["x"] == 1
    assert cs_dict["y"] == -1.0
    assert cs_dict["z"] == 2.0
    assert cs_dict["reference_cs"] is None


def test_coordinate_system_from_dict(set_test_backend):
    """
    Tests the deserialization of a CoordinateSystem instance from a dictionary.
    """
    cs_dict = {"x": 1, "y": -1, "z": 2, "rx": 0, "ry": 0, "rz": 0, "reference_cs": None}
    cs = CoordinateSystem.from_dict(cs_dict)
    assert cs.x == 1
    assert cs.y == -1.0
    assert cs.z == 2.0
    assert cs.reference_cs is None