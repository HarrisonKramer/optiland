import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays


def test_coordinate_system_init():
    # Test case 1: Initialize with default values
    cs = CoordinateSystem()
    assert cs.x == 0
    assert cs.y == 0
    assert cs.z == 0
    assert cs.rx == 0
    assert cs.ry == 0
    assert cs.rz == 0
    assert cs.reference_cs is None

    # Test case 2: Initialize with custom values
    ref_cs = CoordinateSystem(1, 2, 3, 0.1, 0.2, 0.3)
    cs = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, ref_cs)
    assert cs.x == 10
    assert cs.y == 20
    assert cs.z == 30
    assert cs.rx == 0.5
    assert cs.ry == 0.6
    assert cs.rz == 0.7
    assert cs.reference_cs == ref_cs


def test_coordinate_system_localize():
    # Test case 1: Localize rays without reference coordinate system
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    rays = RealRays(1, 2, 5, 0, 0, 1, 1, 1)
    cs.localize(rays)
    assert rays.x == 0.0
    assert rays.y == 3.0
    assert rays.z == 3.0
    assert rays.L == 0.0
    assert rays.M == 0.0
    assert rays.N == 1.0

    # Test case 2: Localize rays with reference coordinate system
    ref_cs = CoordinateSystem(5, 5, 5, 0.2, 0.3, 0.4)
    cs = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, ref_cs)
    rays = RealRays(1, 2, 3, 0.1, 0.2, 0.3, 1, 1)
    cs.localize(rays)
    assert rays.x == pytest.approx(-23.63610642, abs=1e-8)
    assert rays.y == pytest.approx(-25.40225528, abs=1e-8)
    assert rays.z == pytest.approx(-23.08369058, abs=1e-8)
    assert pytest.approx(0.23129557, abs=1e-8) == rays.L
    assert pytest.approx(0.2370124, abs=1e-8) == rays.M
    assert pytest.approx(0.17414787, abs=1e-8) == rays.N


def test_coordinate_system_globalize():
    # Test case 1: Globalize rays without reference coordinate system
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    rays = RealRays(0.0, 3.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    cs.globalize(rays)
    assert rays.x == 1.0
    assert rays.y == 2.0
    assert rays.z == 5.0
    assert rays.L == 0.0
    assert rays.M == 0.0
    assert rays.N == 1.0

    # Test case 2: Globalize rays with reference coordinate system
    ref_cs = CoordinateSystem(5, 5, 5, 0.2, 0.3, 0.4)
    cs = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, ref_cs)
    rays = RealRays(
        -23.63610642,
        -25.40225528,
        -23.08369058,
        0.23129557,
        0.2370124,
        0.17414787,
        1,
        1,
    )
    cs.globalize(rays)
    assert rays.x == pytest.approx(1.0, abs=1e-8)
    assert rays.y == pytest.approx(2.0, abs=1e-8)
    assert rays.z == pytest.approx(3.0, abs=1e-8)
    assert pytest.approx(0.1, abs=1e-8) == rays.L
    assert pytest.approx(0.2, abs=1e-8) == rays.M
    assert pytest.approx(0.3, abs=1e-8) == rays.N


def test_coordinate_system_transform():
    cs1 = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    cs2 = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, cs1)

    eff_translation, eff_rot_mat = cs2.get_effective_transform()
    assert np.allclose(eff_translation, np.array([11, 19, 32]))
    rot_mat = np.array(
        [
            [0.6312515, -0.35830835, 0.68784931],
            [0.5316958, 0.84560449, -0.04746188],
            [-0.56464247, 0.39568697, 0.72430014],
        ],
    )
    assert np.allclose(eff_rot_mat, rot_mat)


def test_coordinate_system_to_dict():
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    cs_dict = cs.to_dict()
    assert cs_dict["x"] == 1
    assert cs_dict["y"] == -1.0
    assert cs_dict["z"] == 2.0
    assert cs_dict["rx"] == 0.0
    assert cs_dict["ry"] == 0.0
    assert cs_dict["rz"] == 0.0
    assert cs_dict["reference_cs"] is None


def test_coordinate_system_from_dict():
    cs_dict = {"x": 1, "y": -1, "z": 2, "rx": 0, "ry": 0, "rz": 0, "reference_cs": None}
    cs = CoordinateSystem.from_dict(cs_dict)
    assert cs.x == 1
    assert cs.y == -1.0
    assert cs.z == 2.0
    assert cs.rx == 0.0
    assert cs.ry == 0.0
    assert cs.rz == 0.0
    assert cs.reference_cs is None
