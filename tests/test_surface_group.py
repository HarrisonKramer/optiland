import pytest
import optiland.backend as be
from optiland.surfaces import SurfaceGroup, StandardSurface, ObjectSurface
from optiland.surfaces.factories import SurfaceFactory # Needed to get CSFactory if not directly importing
from optiland.materials import Material
from optiland.geometries import StandardGeometry
from optiland.coordinate_system import CoordinateSystem

@pytest.fixture
def surface_group():
    return SurfaceGroup()

@pytest.fixture
def object_surface_params():
    # Using StandardGeometry for simplicity, real ObjectSurface might have different geometry needs
    return {
        "surface_type": "ObjectSurface",
        "geometry_params": {"radius": be.inf},
        "material_name": "air",
        "thickness": 10.0, # Default thickness for object surface
        "comment": "Object"
    }

@pytest.fixture
def standard_surface_params():
    return {
        "surface_type": "StandardSurface",
        "geometry_params": {"radius": 50.0},
        "material_name": "N-BK7",
        "thickness": 5.0, # Default thickness
        "comment": "Standard"
    }

class TestSurfaceGroup:
    def test_add_surface_simple_append(self, surface_group, object_surface_params, standard_surface_params):
        # Add Object Surface first (required at index 0)
        surface_group.add_surface(
            surface_type="ObjectSurface", index=0, material=object_surface_params["material_name"],
            thickness=object_surface_params["thickness"], comment="Obj", radius=be.inf
        )
        assert len(surface_group.surfaces) == 1
        assert isinstance(surface_group.surfaces[0], ObjectSurface)
        assert surface_group.surfaces[0].thickness == 10.0

        # Append Standard Surface
        surface_group.add_surface(
            surface_type="StandardSurface", material=standard_surface_params["material_name"],
            thickness=standard_surface_params["thickness"], comment="S1", radius=50.0
            # Implicitly appends if index is None, but our code now requires index if new_surface is None
            # Let's test explicit append via index
            , index=len(surface_group.surfaces)
        )
        assert len(surface_group.surfaces) == 2
        assert isinstance(surface_group.surfaces[1], StandardSurface)
        assert surface_group.surfaces[1].thickness == 5.0
        # Check positions (Object at -10, S1 at 0)
        assert surface_group.surfaces[0].geometry.cs.z == -10.0
        assert surface_group.surfaces[1].geometry.cs.z == 0.0 # Obj.thickness was 10, so Obj is at -10, S1 is at -10 + 10 = 0

    def test_add_object_surface_at_index_0(self, surface_group, object_surface_params):
        surface_group.add_surface(
            surface_type="ObjectSurface", index=0, material=object_surface_params["material_name"],
            thickness=20.0, comment="Obj1", radius=be.inf
        )
        assert isinstance(surface_group.surfaces[0], ObjectSurface)
        assert surface_group.surfaces[0].comment == "Obj1"
        assert surface_group.surfaces[0].thickness == 20.0

        # Replace Object Surface
        surface_group.add_surface(
            surface_type="ObjectSurface", index=0, material=object_surface_params["material_name"],
            thickness=30.0, comment="Obj2", radius=be.inf
        )
        assert len(surface_group.surfaces) == 1
        assert isinstance(surface_group.surfaces[0], ObjectSurface)
        assert surface_group.surfaces[0].comment == "Obj2"
        assert surface_group.surfaces[0].thickness == 30.0

    def test_fail_add_standard_surface_at_index_0(self, surface_group, standard_surface_params):
        with pytest.raises(ValueError, match="Surface at index 0 must be an ObjectSurface"):
            surface_group.add_surface(
                surface_type="StandardSurface", index=0, material=standard_surface_params["material_name"],
                thickness=5.0, radius=100.0
            )

    def test_index_validation_negative(self, surface_group):
        with pytest.raises(ValueError, match="Surface index cannot be negative"):
            surface_group.add_surface(surface_type="StandardSurface", index=-1, thickness=1)

    def test_index_validation_type_error(self, surface_group):
        with pytest.raises(TypeError, match="Surface index must be an integer"):
            surface_group.add_surface(surface_type="StandardSurface", index="1", thickness=1)

    def test_index_validation_out_of_bounds_gap(self, surface_group):
        surface_group.add_surface(surface_type="ObjectSurface", index=0, thickness=10, radius=be.inf)
        with pytest.raises(ValueError, match="Surface index 2 is out of bounds"):
            surface_group.add_surface(surface_type="StandardSurface", index=2, thickness=1) # Gap

    def test_index_validation_insert_at_end_valid(self, surface_group):
        surface_group.add_surface(surface_type="ObjectSurface", index=0, thickness=10, radius=be.inf)
        surface_group.add_surface(surface_type="StandardSurface", index=1, thickness=5, radius=50)
        # Insert another at index 2 (end of list)
        surface_group.add_surface(surface_type="StandardSurface", index=2, thickness=7, radius=60)
        assert len(surface_group.surfaces) == 3
        assert surface_group.surfaces[2].thickness == 7

    def test_out_of_order_insertion_and_positioning(self, surface_group):
        # 0: Object, thickness 10 (to S1) -> Obj at z=-10
        surface_group.add_surface(surface_type="ObjectSurface", index=0, thickness=10, radius=be.inf, comment="Obj")
        # 2 (becomes 1): S2, thickness 7 (to S3), radius 60. Prev is Obj (idx 0).
        # Obj (z=-10) + Obj.thickness (10) = S2 at z=0
        surface_group.add_surface(surface_type="StandardSurface", index=1, thickness=7, radius=60, comment="S2")

        # 1 (insert between Obj and S2): S1, thickness 5 (to S2), radius 50. Prev is Obj (idx 0)
        # Obj (z=-10) + Obj.thickness (10) = S1 at z=0
        # S2 needs to be re-evaluated based on S1.thickness.
        surface_group.add_surface(surface_type="StandardSurface", index=1, thickness=5, radius=50, comment="S1")

        assert len(surface_group.surfaces) == 3
        s_obj = surface_group.surfaces[0]
        s_s1 = surface_group.surfaces[1]
        s_s2 = surface_group.surfaces[2]

        assert s_obj.comment == "Obj"
        assert s_s1.comment == "S1"
        assert s_s2.comment == "S2"

        assert s_obj.thickness == 10 # To S1
        assert s_s1.thickness == 5  # To S2
        assert s_s2.thickness == 7  # To image/next

        # Expected z positions:
        # Obj: z = -Obj.thickness (if finite obj dist for positioning obj surf relative to first surf S1)
        # Let's assume object surface thickness means distance to first optical surface.
        # If Obj.thickness = 10, then Obj is at z = -10 (relative to origin where S1 would be if Obj was at inf)
        # OR, if Obj.thickness means distance from Obj to S1, and S1 is at z=0, then Obj is at z=-10.
        # The CS factory for index 0 does: z_coord = -current_surface_thickness_for_next_surface
        # So s_obj.geometry.cs.z = -s_obj.thickness (if this thickness means obj distance)
        # Let's re-verify CSFactory for index 0: z = -kwargs.get("thickness",0)
        # This 'thickness' is s_obj.thickness. So s_obj.geometry.cs.z = -10.0

        # S1 is at index 1. Prev is s_obj.
        # s_s1.geometry.cs.z = s_obj.geometry.cs.z + s_obj.thickness
        # s_s1.geometry.cs.z = -10.0 + 10.0 = 0.0

        # S2 is at index 2. Prev is s_s1.
        # s_s2.geometry.cs.z = s_s1.geometry.cs.z + s_s1.thickness
        # s_s2.geometry.cs.z = 0.0 + 5.0 = 5.0

        assert be.isclose(s_obj.geometry.cs.z, -10.0)
        assert be.isclose(s_s1.geometry.cs.z, 0.0)
        assert be.isclose(s_s2.geometry.cs.z, 5.0)

    def test_is_stop_handling(self, surface_group):
        surface_group.add_surface(surface_type="ObjectSurface", index=0, thickness=10, radius=be.inf)
        surface_group.add_surface(surface_type="StandardSurface", index=1, thickness=5, radius=50, is_stop=True, comment="S1")
        surface_group.add_surface(surface_type="StandardSurface", index=2, thickness=8, radius=70, comment="S2")

        assert surface_group.surfaces[1].is_stop == True
        assert surface_group.surfaces[0].is_stop == False
        assert surface_group.surfaces[2].is_stop == False

        # Add a new stop surface
        surface_group.add_surface(surface_type="StandardSurface", index=3, thickness=4, radius=30, is_stop=True, comment="S3")
        assert surface_group.surfaces[3].is_stop == True
        assert surface_group.surfaces[1].is_stop == False # Old stop should be cleared

        # Replace a surface with a new stop
        surface_group.add_surface(surface_type="StandardSurface", index=2, thickness=6, radius=-60, is_stop=True, comment="S2_new_stop")
        assert surface_group.surfaces[2].is_stop == True
        assert surface_group.surfaces[2].comment == "S2_new_stop"
        assert surface_group.surfaces[3].is_stop == False # Old stop S3 should be cleared
        assert surface_group.surfaces[1].is_stop == False
