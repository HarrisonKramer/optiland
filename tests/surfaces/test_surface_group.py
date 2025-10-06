# tests/surfaces/test_surface_group.py
"""
Tests for the SurfaceGroup class in optiland.surfaces.surface_group.
"""
import pytest

import optiland.backend as be
from optiland import optic
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import StandardGeometry
from optiland.materials import IdealMaterial
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.surface_group import SurfaceGroup
from ..utils import assert_allclose


def create_real_surface(
    name="s",
    thickness_val=10.0,
    initial_z=0.0,
    is_stop=False,
    comment="",
    radius=be.inf,
    material_n=1.0,
):
    """
    Helper function to create a realistic Surface instance for testing.

    Args:
        name (str): A name for the surface for easier identification in tests.
        thickness_val (float): The thickness to assign to the surface.
        initial_z (float): The initial z-coordinate for the surface.
        is_stop (bool): Whether the surface is the aperture stop.
        comment (str): A comment for the surface.
        radius (float): The radius of curvature of the surface geometry.
        material_n (float): The refractive index for the material.

    Returns:
        A Surface instance with the specified properties.
    """
    cs = CoordinateSystem(z=initial_z)
    geom = StandardGeometry(coordinate_system=cs, radius=radius)
    mat_pre = IdealMaterial(n=material_n)
    mat_post = IdealMaterial(n=material_n)

    surface = Surface(
        geometry=geom,
        material_pre=mat_pre,
        material_post=mat_post,
        is_stop=is_stop,
        comment=comment or name,
        surface_type="Standard",
    )
    surface.thickness = thickness_val
    surface.name = name
    return surface


class TestSurfaceGroupUpdatesRealObjects:
    """
    Tests how the SurfaceGroup handles adding and removing surfaces, with a
    focus on the automatic updates to coordinate systems.
    """

    def _setup_surface_group(
        self,
        num_initial_surfaces=0,
        use_absolute_cs=False,
        thicknesses=None,
        initial_zs=None,
    ):
        """
        Helper to create a SurfaceGroup with a specified number of real surfaces
        and coordinate system mode.
        """
        sg = SurfaceGroup([])
        sg.surface_factory.use_absolute_cs = use_absolute_cs

        initial_surfaces = []
        if thicknesses is None:
            thicknesses = [10.0] * num_initial_surfaces
        if initial_zs is None:
            initial_zs = (
                [-100.0 if i == 0 else 0.0 for i in range(num_initial_surfaces)]
                if num_initial_surfaces > 0
                else []
            )

        for i in range(num_initial_surfaces):
            surface = create_real_surface(
                name=f"s{i}", thickness_val=thicknesses[i], initial_z=initial_zs[i]
            )
            initial_surfaces.append(surface)

        sg.surfaces = initial_surfaces

        if not use_absolute_cs and len(sg.surfaces) > 1:
            sg._update_coordinate_systems(start_index=0)
        return sg

    @pytest.mark.parametrize("use_absolute_cs", [True, False])
    def test_add_surface_new_object_append_empty(
        self, set_test_backend, use_absolute_cs
    ):
        """
        Tests adding a surface to an empty SurfaceGroup. The surface should be
        appended and retain its initial z-coordinate.
        """
        sg = self._setup_surface_group(use_absolute_cs=use_absolute_cs)
        new_surf = create_real_surface(name="new", thickness_val=5.0, initial_z=10.0)
        sg.add_surface(new_surface=new_surf)

        assert len(sg.surfaces) == 1
        assert sg.surfaces[0] is new_surf
        assert_allclose(sg.surfaces[0].geometry.cs.z, be.array(10.0))

    def test_add_surface_new_object_append_non_empty_abs_cs(self, set_test_backend):
        """
        Tests appending a surface to a non-empty group in absolute coordinate
        mode. The existing surface's position should not change, and the new
        surface should retain its initial z-coordinate.
        """
        sg = self._setup_surface_group(
            num_initial_surfaces=1,
            use_absolute_cs=True,
            thicknesses=[10],
            initial_zs=[5.0],
        )
        s0_z = be.copy(sg.surfaces[0].geometry.cs.z)

        new_surf = create_real_surface(name="new", thickness_val=5.0, initial_z=20.0)
        sg.add_surface(new_surface=new_surf)

        assert len(sg.surfaces) == 2
        assert sg.surfaces[1] is new_surf
        assert_allclose(sg.surfaces[0].geometry.cs.z, s0_z)
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(20.0))

    def test_add_surface_new_object_append_non_empty_rel_cs_no_update(
        self, set_test_backend
    ):
        """
        Tests appending a surface in relative coordinate mode. No coordinate
        update should be triggered, and the new surface should retain its
        initial z-coordinate.
        """
        sg = self._setup_surface_group(
            num_initial_surfaces=1,
            use_absolute_cs=False,
            thicknesses=[10],
            initial_zs=[-100.0],
        )
        s0 = sg.surfaces[0]
        new_surf = create_real_surface(
            name="S1_appended", thickness_val=5.0, initial_z=123.0
        )
        sg.add_surface(new_surface=new_surf)

        assert len(sg.surfaces) == 2
        assert sg.surfaces[1] is new_surf
        assert_allclose(s0.geometry.cs.z, be.array(-100.0))
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(123.0))

    def test_add_surface_new_object_insert_middle_rel_cs_triggers_update(
        self, set_test_backend
    ):
        """
        Tests inserting a surface into the middle of a group in relative
        coordinate mode. This should trigger a coordinate update for all
        subsequent surfaces.
        """
        sg = self._setup_surface_group(
            num_initial_surfaces=2,
            use_absolute_cs=False,
            thicknesses=[10, 20],
            initial_zs=[-100.0, 0.0],
        )
        s_new = create_real_surface(name="S_new", thickness_val=5.0, initial_z=123.0)
        sg.add_surface(new_surface=s_new, index=1)

        assert len(sg.surfaces) == 3
        assert sg.surfaces[1] is s_new
        assert_allclose(sg.surfaces[0].geometry.cs.z, be.array(-100.0))
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(0.0))
        assert_allclose(sg.surfaces[2].geometry.cs.z, be.array(s_new.thickness))

    def test_add_surface_new_object_is_stop_propagation(self, set_test_backend):
        """
        Tests that when a new surface is added with `is_stop=True`, the stop
        flag is correctly cleared from the previously designated stop surface.
        """
        sg = self._setup_surface_group(num_initial_surfaces=2, use_absolute_cs=False)
        sg.surfaces[0].is_stop = True
        new_surf = create_real_surface(name="new_stop", is_stop=True)
        sg.add_surface(new_surface=new_surf, index=len(sg.surfaces))

        assert len(sg.surfaces) == 3
        assert not sg.surfaces[0].is_stop
        assert not sg.surfaces[1].is_stop
        assert sg.surfaces[2].is_stop

    def test_add_surface_by_creation_rel_cs(self, set_test_backend):
        """
        Tests adding surfaces by creation using the factory, verifying that
        insertions trigger coordinate updates correctly in relative mode.
        """
        sg = self._setup_surface_group(
            num_initial_surfaces=1,
            use_absolute_cs=False,
            thicknesses=[10],
            initial_zs=[-100.0],
        )
        sg.add_surface(
            surface_type="standard", index=1, thickness=5.0, comment="S1_created"
        )

        assert len(sg.surfaces) == 2
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(0.0))

        sg.add_surface(
            surface_type="standard", index=1, thickness=12.0, comment="S2_inserted"
        )

        assert len(sg.surfaces) == 3
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(0.0))
        assert_allclose(sg.surfaces[2].geometry.cs.z, be.array(12.0))

    def test_add_surface_by_creation_error_no_index(self, set_test_backend):
        """
        Tests that creating a surface without specifying an index raises a
        ValueError.
        """
        sg = self._setup_surface_group(use_absolute_cs=False)
        with pytest.raises(
            ValueError, match="Must define index when defining surface."
        ):
            sg.add_surface(surface_type="standard", comment="no_index_surf")

    def test_remove_surface_middle_rel_cs_triggers_update(self, set_test_backend):
        """
        Tests that removing a surface from the middle of a group in relative
        mode triggers a coordinate update for subsequent surfaces.
        """
        sg = self._setup_surface_group(
            num_initial_surfaces=3,
            use_absolute_cs=False,
            thicknesses=[10, 20, 30],
            initial_zs=[-100, 0, 20],
        )
        sg.remove_surface(index=1)

        assert len(sg.surfaces) == 2
        assert_allclose(sg.surfaces[0].geometry.cs.z, be.array(-100.0))
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(0.0))

    def test_remove_surface_last_rel_cs_no_update_triggered(self, set_test_backend):
        """
        Tests that removing the last optical surface in relative mode does not
        trigger a coordinate update.
        """
        sg = self._setup_surface_group(
            num_initial_surfaces=3,
            use_absolute_cs=False,
            thicknesses=[10, 20, 30],
            initial_zs=[-100, 0, 20],
        )
        s1_z_before_removal = be.copy(sg.surfaces[1].geometry.cs.z)
        sg.remove_surface(index=2)

        assert len(sg.surfaces) == 2
        assert_allclose(sg.surfaces[1].geometry.cs.z, s1_z_before_removal)

    def test_remove_surface_abs_cs_no_update(self, set_test_backend):
        """
        Tests that removing a surface in absolute mode does not trigger a
        coordinate update.
        """
        sg = self._setup_surface_group(num_initial_surfaces=3, use_absolute_cs=True)
        sg.surfaces[0].geometry.cs.z = be.array(0.0)
        sg.surfaces[1].geometry.cs.z = be.array(10.0)
        sg.surfaces[2].geometry.cs.z = be.array(30.0)
        s2_z_original = be.copy(sg.surfaces[2].geometry.cs.z)

        sg.remove_surface(index=1)

        assert len(sg.surfaces) == 2
        assert_allclose(sg.surfaces[1].geometry.cs.z, s2_z_original)

    def test_remove_surface_error_remove_object(self, set_test_backend):
        """
        Tests that attempting to remove the object surface (index 0) raises
        a ValueError.
        """
        sg = self._setup_surface_group(num_initial_surfaces=1)
        with pytest.raises(ValueError, match="Cannot remove object surface"):
            sg.remove_surface(index=0)

    def test_add_surface_new_object_error_negative_index(self, set_test_backend):
        """
        Tests that adding a surface with a negative index raises an IndexError.
        """
        sg = self._setup_surface_group(num_initial_surfaces=1)
        with pytest.raises(IndexError):
            sg.add_surface(new_surface=create_real_surface(), index=-1)

    def test_add_surface_new_object_error_index_out_of_bounds(self, set_test_backend):
        """
        Tests that adding a surface at an index greater than the number of
        surfaces raises an IndexError.
        """
        sg = self._setup_surface_group(num_initial_surfaces=1)
        with pytest.raises(IndexError):
            sg.add_surface(new_surface=create_real_surface(), index=2)

    def test_add_surface_by_creation_error_negative_index(self, set_test_backend):
        """
        Tests that creating a surface with a negative index raises an IndexError.
        """
        sg = self._setup_surface_group()
        with pytest.raises(IndexError):
            sg.add_surface(surface_type="standard", index=-1, thickness=1)

    def test_add_surface_by_creation_error_index_out_of_bounds(self, set_test_backend):
        """
        Tests that creating a surface at an index greater than the number of
        surfaces raises an IndexError.
        """
        sg = self._setup_surface_group(num_initial_surfaces=1)
        with pytest.raises(IndexError):
            sg.add_surface(surface_type="standard", index=2, thickness=1)

    def test_remove_surface_error_index_out_of_bounds_negative(self, set_test_backend):
        """
        Tests that removing a surface with a negative index raises an IndexError.
        """
        sg = self._setup_surface_group(num_initial_surfaces=2)
        with pytest.raises(IndexError):
            sg.remove_surface(index=-1)

    def test_remove_surface_error_index_out_of_bounds_too_large(self, set_test_backend):
        """
        Tests that removing a surface at an index equal to or greater than the
        number of surfaces raises an IndexError.
        """
        sg = self._setup_surface_group(num_initial_surfaces=2)
        with pytest.raises(IndexError):
            sg.remove_surface(index=2)
        with pytest.raises(IndexError):
            sg.remove_surface(index=3)

    def test_update_coordinate_systems_infinite_thickness_error(self, set_test_backend):
        """
        Tests that attempting to update coordinates after a surface with
        infinite thickness raises a ValueError.
        """
        sg = self._setup_surface_group(
            num_initial_surfaces=2,
            use_absolute_cs=False,
            thicknesses=[10, be.inf],
            initial_zs=[-100, 0],
        )
        s2 = create_real_surface(name="s2_after_inf", thickness_val=5.0)
        sg.surfaces.append(s2)
        with pytest.raises(ValueError):
            sg._update_coordinate_systems(start_index=2)

    def test_update_coordinate_systems_thickness_is_be_array(self, set_test_backend):
        """
        Tests that the coordinate update works correctly when a surface's
        thickness is a backend array.
        """
        sg = self._setup_surface_group(
            num_initial_surfaces=3,
            use_absolute_cs=False,
            thicknesses=[10, 20, 30],
            initial_zs=[-100, 0, 20],
        )
        sg.surfaces[1].thickness = be.array(25.0)
        sg._update_coordinate_systems(start_index=2)
        assert_allclose(sg.surfaces[2].geometry.cs.z, be.array(25.0))

    def test_insert_all_at_index_1(self, set_test_backend):
        """
        A regression test that constructs a Cooke triplet by repeatedly
        inserting surfaces at index 1. This verifies that a complex sequence
        of insertions and coordinate updates produces a valid, traceable system.
        """
        lens = optic.Optic()

        lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
        lens.add_surface(index=1)
        lens.add_surface(index=1, radius=-18.39533, thickness=42.20778)
        lens.add_surface(index=1, radius=79.68360, thickness=2.95208, material="SK16")
        lens.add_surface(index=1, radius=20.29192, thickness=4.75041, is_stop=True)
        lens.add_surface(
            index=1, radius=-22.21328, thickness=0.99997, material=("F2", "schott")
        )
        lens.add_surface(index=1, radius=-435.76044, thickness=6.00755)
        lens.add_surface(index=1, radius=22.01359, thickness=3.25896, material="SK16")

        lens.set_aperture(aperture_type="EPD", value=10)
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        lens.add_field(y=14)
        lens.add_field(y=20)
        lens.add_wavelength(value=0.48)
        lens.add_wavelength(value=0.55, is_primary=True)
        lens.add_wavelength(value=0.65)

        rays = lens.trace(
            Hx=0, Hy=1, distribution="hexapolar", num_rays=3, wavelength=0.59
        )
        assert_allclose(be.mean(rays.y), 3.47484521)