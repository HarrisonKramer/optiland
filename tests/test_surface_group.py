import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import StandardGeometry
from optiland.materials import IdealMaterial
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.surface_group import SurfaceGroup

from tests.utils import assert_allclose


# Helper to create a real Surface instance for tests
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
    Creates a real Surface object.
    The 'thickness' attribute is crucial for SurfaceGroup._update_coordinate_systems.
    We will set it directly on the surface instance.
    """
    cs = CoordinateSystem(z=initial_z)
    # StandardGeometry needs a radius; be.inf can represent a plane.
    geom = StandardGeometry(coordinate_system=cs, radius=radius)

    # Surfaces require material_pre and material_post
    mat_pre = IdealMaterial(n=material_n)
    mat_post = IdealMaterial(n=material_n)  # Keep it simple for now, or vary if needed

    surface = Surface(
        geometry=geom,
        material_pre=mat_pre,
        material_post=mat_post,
        is_stop=is_stop,
        comment=comment or name,
        surface_type="Standard",  # A default type
    )
    surface.thickness = thickness_val
    surface.name = name  # add for testing

    return surface


class TestSurfaceGroupUpdatesRealObjects:
    def _setup_surface_group(
        self,
        num_initial_surfaces=0,
        use_absolute_cs=False,
        thicknesses=None,
        initial_zs=None,
    ):
        """Helper to create a SurfaceGroup with real surfaces."""
        sg = SurfaceGroup([])  # Start with empty
        # The SurfaceGroup initializes its own SurfaceFactory.
        # We can configure its 'use_absolute_cs' property.
        sg.surface_factory.use_absolute_cs = use_absolute_cs

        initial_surfaces = []
        if thicknesses is None:
            thicknesses = [10.0] * num_initial_surfaces
        if initial_zs is None:
            # Default initial Z: obj at -100, others at 0 before potential update
            initial_zs = [
                -100.0 if i == 0 else 0.0 for i in range(num_initial_surfaces)
            ]
            if num_initial_surfaces == 0:
                initial_zs = []

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
        sg = self._setup_surface_group(use_absolute_cs=use_absolute_cs)
        new_surf = create_real_surface(name="new", thickness_val=5.0, initial_z=10.0)
        sg.add_surface(new_surface=new_surf)

        assert len(sg.surfaces) == 1
        assert sg.surfaces[0] is new_surf
        assert_allclose(
            sg.surfaces[0].geometry.cs.z, be.array(10.0)
        )  # Keeps its initial Z

    def test_add_surface_new_object_append_non_empty_abs_cs(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=1,
            use_absolute_cs=True,
            thicknesses=[10],
            initial_zs=[5.0],
        )
        s0_z = be.copy(sg.surfaces[0].geometry.cs.z)

        new_surf = create_real_surface(name="new", thickness_val=5.0, initial_z=20.0)
        sg.add_surface(new_surface=new_surf)  # Appends

        assert len(sg.surfaces) == 2
        assert sg.surfaces[1] is new_surf
        assert_allclose(sg.surfaces[0].geometry.cs.z, s0_z)  # Unchanged
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(20.0))  # Keeps its Z

    def test_add_surface_new_object_append_non_empty_rel_cs_no_update(
        self, set_test_backend
    ):
        sg = self._setup_surface_group(
            num_initial_surfaces=1,
            use_absolute_cs=False,
            thicknesses=[10],
            initial_zs=[-100.0],
        )
        # S0(z=-100, t=10)
        s0 = sg.surfaces[0]

        new_surf = create_real_surface(
            name="S1_appended", thickness_val=5.0, initial_z=123.0
        )
        # Appending: index is not specified or index == len(sg.surfaces)
        sg.add_surface(
            new_surface=new_surf
        )  # Appends, update_start_index points to new_surf, is_last_surface=True

        assert len(sg.surfaces) == 2
        assert sg.surfaces[1] is new_surf
        assert_allclose(s0.geometry.cs.z, be.array(-100.0))
        # _update_coordinate_systems is NOT called when appending to make it the last surface
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(123.0))

    def test_add_surface_new_object_insert_middle_rel_cs_triggers_update(
        self, set_test_backend
    ):
        sg = self._setup_surface_group(
            num_initial_surfaces=2,
            use_absolute_cs=False,
            thicknesses=[10, 20],
            initial_zs=[-100.0, 0.0],
        )
        # After setup: S0(z=-100, t=10), S1_orig(z=0, t=20)
        s1_orig_name = sg.surfaces[1].name

        s_new = create_real_surface(
            name="S_new", thickness_val=5.0, initial_z=123.0
        )  # initial_z overridden
        sg.add_surface(new_surface=s_new, index=1)

        assert len(sg.surfaces) == 2
        assert sg.surfaces[1] is s_new
        assert sg.surfaces[0].name == "s0"

        assert_allclose(
            sg.surfaces[0].geometry.cs.z, be.array(-100.0)
        )  # S0 z unchanged
        assert_allclose(
            sg.surfaces[1].geometry.cs.z, be.array(123.0)
        )

    def test_add_surface_new_object_is_stop_propagation(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=2, use_absolute_cs=False)
        sg.surfaces[0].is_stop = True  # Set S0 as stop
        new_surf = create_real_surface(name="new_stop", is_stop=True)
        sg.add_surface(new_surface=new_surf, index=2)  # Append

        assert len(sg.surfaces) == 3
        assert not sg.surfaces[0].is_stop  # Old stop is cleared
        assert not sg.surfaces[1].is_stop
        assert sg.surfaces[2].is_stop

    # --- Tests for add_surface by creation (new_surface is None) ---
    def test_add_surface_by_creation_rel_cs(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=1,
            use_absolute_cs=False,
            thicknesses=[10],
            initial_zs=[-100.0],
        )

        # Add S1 by creation
        # The 'thickness' kwarg should be handled by the actual SurfaceFactory
        sg.add_surface(
            surface_type="standard",
            comment="S1_created",
            index=1,
            material="air",
            thickness=5.0,
        )  # 'air' is just a placeholder for material name

        assert len(sg.surfaces) == 2
        created_s1 = sg.surfaces[1]
        assert created_s1.comment == "S1_created"
        assert created_s1.thickness == 5.0  # Assuming factory sets this

        # Add S1 by creation
        sg.add_surface(
            surface_type="standard",
            comment="S2_inserted",
            index=1,
            material="glass",
            thickness=12.0,
        )
        # Surfaces: S0, S2_inserted

        assert len(sg.surfaces) == 2
        s1_created_now_s2 = sg.surfaces[1]

        assert s1_created_now_s2.comment == "S2_inserted"
        assert s1_created_now_s2.thickness == 12.0

    def test_add_surface_by_creation_error_no_index(self, set_test_backend):
        sg = self._setup_surface_group(use_absolute_cs=False)
        with pytest.raises(
            ValueError, match="Must define index when defining surface."
        ):
            sg.add_surface(surface_type="standard", comment="no_index_surf")

    # --- Tests for remove_surface ---
    def test_remove_surface_middle_rel_cs_triggers_update(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=3,
            use_absolute_cs=False,
            thicknesses=[10, 20, 30],
            initial_zs=[-100, 0, 20],
        )
        # Initial: S0(z=-100,t=10), S1(z=0,t=20), S2(z=20,t=30)
        s2_original_comment = sg.surfaces[2].comment  # Was s2

        sg.remove_surface(index=1)  # Remove S1

        assert len(sg.surfaces) == 2
        assert sg.surfaces[0].comment == "s0"
        assert sg.surfaces[1].comment == s2_original_comment  # Old S2 is now S1

        assert_allclose(
            sg.surfaces[0].geometry.cs.z, be.array(-100.0)
        )  # S0 unchanged
        # Old S2 (now new S1) z should be 0.0 because it's now the surface at index 1 and update was called
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(0.0))

    def test_remove_surface_last_rel_cs_no_update_triggered(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=3,
            use_absolute_cs=False,
            thicknesses=[10, 20, 30],
            initial_zs=[-100, 0, 20],
        )
        # S0(z=-100,t=10), S1(z=0,t=20), S2(z=20,t=30)
        s1_z_before_removal = be.copy(sg.surfaces[1].geometry.cs.z)

        sg.remove_surface(index=2)  # Remove S2 (last optical surface)
        # _update_coordinate_systems not called because was_not_last_surface is false

        assert len(sg.surfaces) == 2
        assert_allclose(
            sg.surfaces[1].geometry.cs.z, s1_z_before_removal
        )  # S1 z unchanged

    def test_remove_surface_abs_cs_no_update(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=3, use_absolute_cs=True)
        # Set specific Zs that wouldn't occur with relative CS updates
        sg.surfaces[0].geometry.cs.z = be.array(0.0)
        sg.surfaces[1].geometry.cs.z = be.array(10.0)
        sg.surfaces[2].geometry.cs.z = be.array(30.0)

        s2_z_original = be.copy(sg.surfaces[2].geometry.cs.z)

        sg.remove_surface(index=1)  # Remove S1

        assert len(sg.surfaces) == 2
        # With absolute_cs=True, _update_coordinate_systems is not called
        assert_allclose(
            sg.surfaces[1].geometry.cs.z, s2_z_original
        )  # Old S2 (now S1) keeps its original Z

    def test_remove_surface_error_remove_object(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=1)
        with pytest.raises(ValueError, match="Cannot remove object surface"):
            sg.remove_surface(index=0)

    # --- Error condition tests from SurfaceGroup code directly ---
    def test_add_surface_new_object_error_negative_index(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=1)
        with pytest.raises(IndexError, match="Index -1 cannot be negative."):
            sg.add_surface(new_surface=create_real_surface(), index=-1)

    def test_add_surface_new_object_error_index_out_of_bounds(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=1)  # surfaces[0] exists
        # Valid indices for insertion/overwrite: 0, 1 (append)
        with pytest.raises(IndexError, match="Index 2 is out of bounds"):
            sg.add_surface(new_surface=create_real_surface(), index=2)

    def test_add_surface_by_creation_error_negative_index(self, set_test_backend):
        sg = self._setup_surface_group()
        with pytest.raises(IndexError):
            sg.add_surface(
                surface_type="standard", index=-1, thickness=1
            )  # Assuming factory handles thickness

    def test_add_surface_by_creation_error_index_out_of_bounds(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=1)
        with pytest.raises(IndexError):
            sg.add_surface(surface_type="standard", index=2, thickness=1)

    def test_remove_surface_error_index_out_of_bounds_negative(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=2)
        with pytest.raises(IndexError, match="Index -1 is out of bounds"):
            sg.remove_surface(index=-1)

    def test_remove_surface_error_index_out_of_bounds_too_large(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=2)  # Valid remove index: 1
        with pytest.raises(IndexError, match="Index 2 is out of bounds"):
            sg.remove_surface(index=2)
        with pytest.raises(
            IndexError, match="Index 3 is out of bounds"
        ):  # Also too large
            sg.remove_surface(index=3)

    # --- Tests for _update_coordinate_systems specific cases ---
    def test_update_coordinate_systems_infinite_thickness_error(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=2,
            use_absolute_cs=False,
            thicknesses=[10, be.inf],
            initial_zs=[-100, 0],
        )
        # S0(z=-100, t=10), S1(z=0, t=inf)
        # Add a third surface; its position calculation will depend on S1's infinite thickness
        s2 = create_real_surface(name="s2_after_inf", thickness_val=5.0)
        sg.surfaces.append(s2)  # Now [S0, S1, s2]

        # Update starting from index 2 (s2_after_inf), which looks at s1's thickness
        with pytest.raises(
            ValueError,
            match="Coordinate system update failed due to infinite thickness at surface 1",
        ):
            sg._update_coordinate_systems(start_index=2)

    def test_update_coordinate_systems_thickness_is_be_array(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=3,
            use_absolute_cs=False,
            thicknesses=[10, 20, 30],
            initial_zs=[-100, 0, 20],
        )
        # S0(z=-100,t=10), S1(z=0,t=20), S2(z=20,t=30)
        sg.surfaces[1].thickness = be.array(25.0)  # S1 thickness is a 0-d array

        sg._update_coordinate_systems(start_index=2)  # Update S2 based on S1

        # S2.z = S1.z + S1.thickness = 0.0 + 25.0
        assert_allclose(sg.surfaces[2].geometry.cs.z, be.array(25.0))
