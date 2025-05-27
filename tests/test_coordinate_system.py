import optiland.backend as be
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays
from .utils import assert_allclose


def test_coordinate_system_init(set_test_backend):
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


def test_coordinate_system_localize(set_test_backend):
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
    assert_allclose(rays.x, -1.826215)
    assert_allclose(rays.y, -26.51361)
    assert_allclose(rays.z, -32.55361)
    assert_allclose(rays.L, -0.0122750)
    assert_allclose(rays.M, 0.2697627)
    assert_allclose(rays.N, 0.2589931)


def test_coordinate_system_globalize(set_test_backend):
    # Test case 1: Globalize rays without reference coordinate system
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    rays = RealRays(0.0, 3.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    cs.globalize(rays)
    assert_allclose(rays.x, 1.0)
    assert_allclose(rays.y, 2.0)
    assert_allclose(rays.z, 5.0)
    assert_allclose(rays.L, 0.0)
    assert_allclose(rays.M, 0.0)
    assert_allclose(rays.N, 1.0)

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
    assert_allclose(rays.x, 4.654692)
    assert_allclose(rays.y, -12.50279)
    assert_allclose(rays.z, 21.51751)
    assert_allclose(rays.L, 0.0866579)
    assert_allclose(rays.M, 0.3526931)
    assert_allclose(rays.N, 0.08998877)


def test_coordinate_system_transform(set_test_backend):
    cs1 = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    cs2 = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, cs1)

    eff_translation, eff_rot_mat = cs2.get_effective_transform()
    assert be.allclose(eff_translation, be.array([11, 19, 32]))
    rot_mat = be.array(
        [
            [0.6312515, -0.35830835, 0.68784931],
            [0.5316958, 0.84560449, -0.04746188],
            [-0.56464247, 0.39568697, 0.72430014],
        ],
    )
    assert be.allclose(eff_rot_mat, rot_mat)


def test_coordinate_system_to_dict(set_test_backend):
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    cs_dict = cs.to_dict()
    assert cs_dict["x"] == 1
    assert cs_dict["y"] == -1.0
    assert cs_dict["z"] == 2.0
    assert cs_dict["rx"] == 0.0
    assert cs_dict["ry"] == 0.0
    assert cs_dict["rz"] == 0.0
    assert cs_dict["reference_cs"] is None


def test_coordinate_system_from_dict(set_test_backend):
    cs_dict = {"x": 1, "y": -1, "z": 2, "rx": 0, "ry": 0, "rz": 0, "reference_cs": None}
    cs = CoordinateSystem.from_dict(cs_dict)
    assert cs.x == 1
    assert cs.y == -1.0
    assert cs.z == 2.0
    assert cs.rx == 0.0
    assert cs.ry == 0.0
    assert cs.rz == 0.0
    assert cs.reference_cs is None
