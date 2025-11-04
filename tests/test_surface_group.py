"""Integration tests for the refactored SurfaceGroup and surface creation process."""

import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.surface_group import SurfaceGroup
from tests.utils import assert_allclose


def test_surface_group_add_object_surface(set_test_backend):
    """Verify adding the initial object surface at index 0."""
    sg = SurfaceGroup()
    sg.add_surface(index=0, comment="Object", thickness=100.0, material="air")

    assert len(sg.surfaces) == 1
    s0 = sg.surfaces[0]
    assert isinstance(s0, ObjectSurface)
    assert s0.comment == "Object"
    assert s0.thickness == 100.0
    assert isinstance(s0.material_post, IdealMaterial)
    assert_allclose(s0.geometry.cs.z, be.array(-100.0))


def test_surface_group_add_standard_surface(set_test_backend):
    """Verify adding a standard surface after the object."""
    sg = SurfaceGroup()
    sg.add_surface(index=0, thickness=100.0, material="air")
    sg.add_surface(
        index=1,
        comment="S1",
        thickness=10.0,
        material=IdealMaterial(n=1.5),
        radius=50.0,
    )

    assert len(sg.surfaces) == 2
    s1 = sg.surfaces[1]
    assert isinstance(s1, Surface)
    assert s1.comment == "S1"
    assert s1.thickness == 10.0
    assert_allclose(s1.geometry.cs.z, be.array(0.0))
    assert s1.material_pre.n(0.5) == 1.0
    assert s1.material_post.n(0.5) == 1.5
    assert s1.geometry.radius == 50.0


def test_surface_group_add_paraxial_surface(set_test_backend):
    """Verify adding a surface with the 'paraxial' type."""
    sg = SurfaceGroup()
    sg.add_surface(index=0, thickness=100.0, material="air")
    sg.add_surface(
        index=1,
        surface_type="paraxial",
        comment="Thin Lens",
        thickness=20.0,
        material="air",
        f=50.0,
    )

    assert len(sg.surfaces) == 2
    s1 = sg.surfaces[1]
    assert s1.interaction_model.__class__.__name__ == "ThinLensInteractionModel"
    assert s1.interaction_model.f == 50.0  # Corrected attribute
    assert_allclose(s1.geometry.cs.z, be.array(0.0))
    assert s1.thickness == 20.0


def test_surface_group_insert_surface_updates_z(set_test_backend):
    """Verify that inserting a surface correctly updates subsequent z-positions."""
    sg = SurfaceGroup()
    sg.add_surface(index=0, thickness=100.0, material="air")
    sg.add_surface(index=1, thickness=10.0, material="BK7")
    sg.add_surface(index=2, thickness=20.0, material="air")

    sg.add_surface(index=2, comment="New S2", thickness=5.0, material="F2")

    assert len(sg.surfaces) == 4
    s0, s1, s_new, s_old_2 = sg.surfaces

    assert s_new.comment == "New S2"
    assert_allclose(s1.geometry.cs.z, be.array(0.0))
    assert_allclose(s_new.geometry.cs.z, be.array(10.0))
    assert_allclose(s_old_2.geometry.cs.z, be.array(15.0))


def test_surface_group_add_mirror_surface(set_test_backend):
    """Verify that a 'mirror' surface correctly inherits the precedent material."""
    sg = SurfaceGroup()
    glass = IdealMaterial(n=1.5)
    sg.add_surface(index=0, thickness=100.0, material=glass)
    sg.add_surface(index=1, thickness=10.0, material="mirror")

    assert len(sg.surfaces) == 2
    mirror_surface = sg.surfaces[1]

    assert mirror_surface.interaction_model.is_reflective is True
    assert mirror_surface.material_pre is glass
    assert mirror_surface.material_post is glass


def test_remove_surface_updates_state_and_links(set_test_backend):
    """Verify removing a surface correctly updates links and coordinates."""
    optic = Optic()
    mat1 = IdealMaterial(n=1.5)
    mat2 = IdealMaterial(n=2.0)

    optic.add_surface(index=0, material="air", thickness=5)
    optic.add_surface(index=1, material=mat1, thickness=10)
    optic.add_surface(index=2, material=mat2, thickness=15)
    optic.add_surface(index=3, material="air", thickness=20)

    s3_before = optic.surface_group.surfaces[3]
    assert s3_before.material_pre is mat2
    assert_allclose(s3_before.geometry.cs.z, be.array(25.0))

    optic.surface_group.remove_surface(2)

    assert len(optic.surface_group.surfaces) == 3
    s3_after = optic.surface_group.surfaces[2]

    assert s3_after.material_pre is mat1
    assert_allclose(s3_after.geometry.cs.z, be.array(10.0))
