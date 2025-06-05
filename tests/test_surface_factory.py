import optiland.backend as be
import pytest

from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.materials import IdealMaterial
from optiland.samples.objectives import TessarLens
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.paraxial_surface import ParaxialSurface
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces import SurfaceFactory
from optiland.geometries.plane import Plane
from optiland.geometries.standard import StandardGeometry


class TestSurfaceFactory:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.lens = TessarLens()
        self.surface_group = self.lens.surface_group
        self.factory = SurfaceFactory(self.surface_group)

    def test_create_surface_standard(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Standard",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_even_asphere(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="even_asphere",
            comment="Even Asphere",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coefficients=[1, 2, 3],
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert surface.geometry.c == [1, 2, 3]
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_odd_asphere(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="odd_asphere",
            comment="Odd Asphere",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coefficients=[1, 2, 3],
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert surface.geometry.c == [1, 2, 3]
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_polynomial(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="polynomial",
            comment="Polynomial",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coefficients=[1, 2, 3],
            tol=1e-6,
            max_iter=100,
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert be.array_equal(surface.geometry.c, be.array([[1, 2, 3]]))
        assert surface.geometry.tol == 1e-6
        assert surface.geometry.max_iter == 100
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_chebyshev(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="chebyshev",
            comment="Chebyshev",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coefficients=be.arange(9).reshape(3, 3),
            tol=1e-6,
            max_iter=100,
            norm_x=1,
            norm_y=1,
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert be.all(surface.geometry.c == be.arange(9).reshape(3, 3))
        assert surface.geometry.tol == 1e-6
        assert surface.geometry.max_iter == 100
        assert surface.geometry.norm_x == 1
        assert surface.geometry.norm_y == 1
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_object(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Object",
            index=0,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
        )
        assert isinstance(surface, ObjectSurface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert isinstance(surface.material_post, IdealMaterial)

    def test_invalid_surface_type(self, set_test_backend):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="invalid",
                comment="Invalid",
                index=1,
                is_stop=False,
                material="air",
                thickness=5,
            )

    def test_create_surface_plane_via_registry(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="plane",
            comment="Plane via Registry",
            index=1,
            is_stop=False,
            material="air",
            thickness=2,
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.geometry, Plane)
        assert surface.comment == "Plane via Registry"

    def test_create_surface_sphere_via_registry(self, set_test_backend):
        radius = 20.0
        surface = self.factory.create_surface(
            surface_type="sphere",
            comment="Sphere via Registry",
            index=1,
            is_stop=False,
            material="glass", # Example material
            thickness=3,
            radius=radius,
        )
        assert isinstance(surface, Surface)
        # "sphere" surface_type uses StandardGeometry
        assert isinstance(surface.geometry, StandardGeometry)
        assert surface.geometry.radius == radius
        assert surface.geometry.k == 0 # For a sphere, conic constant is 0
        assert surface.comment == "Sphere via Registry"

    def test_register_and_create_mock_surface(self, monkeypatch, set_test_backend):
        # 1. Define MockSurface class
        class MockSurface:
            def __init__(self, comment, **kwargs): # Keep it simple
                self.comment = comment
                self.thickness = 0.0 # required attribute

        # 2. Define mock handler function
        # Mimics the signature of other handlers
        @staticmethod
        def mock_handler_function(
            factory_instance, # Unused, but part of signature
            surface_type_str, # Unused
            comment,
            index,            # Unused
            is_stop,          # Unused
            material_pre,     # Unused
            material_post,    # Unused
            is_reflective,    # Unused
            coating,          # Unused
            coordinate_system,# Unused
            geometry_config,  # Unused
            **kwargs,
        ):
            # The handler should return an instance of the surface type
            return MockSurface(comment=comment)

        # 3. Register the mock handler using monkeypatch
        monkeypatch.setitem(SurfaceFactory._surface_registry, "mock_type", mock_handler_function)

        # 4. Call create_surface for the mock_type
        mock_surface_obj = self.factory.create_surface(
            surface_type="mock_type",
            comment="Mock Surface Test",
            index=1, # Needs to be non-zero to avoid ObjectSurface logic
            is_stop=False,
            material="air", # material is needed by factory
            thickness=1 # thickness is set by factory on returned obj
        )

        # 5. Assert the returned object is an instance of MockSurface
        assert isinstance(mock_surface_obj, MockSurface)
        assert mock_surface_obj.comment == "Mock Surface Test"
        assert mock_surface_obj.thickness == 1 # Check thickness is set

    def test_unregistered_surface_type(self, set_test_backend):
        non_existent_type = "nonexistent_surface_xyz_123"
        with pytest.raises(ValueError, match=f"Unknown or unregistered surface_type: {non_existent_type}"):
            self.factory.create_surface(
                surface_type=non_existent_type,
                comment="Unregistered Test",
                index=1,
                is_stop=False,
                material="air",
                thickness=5,
            )

    def test_invalid_surface_index(self, set_test_backend):
        with pytest.raises(IndexError):
            self.factory.create_surface(
                surface_type="standard",
                comment="Invalid",
                index=42,
                is_stop=False,
                material="air",
                thickness=5,
            )

    def test_create_surface_with_coating(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Coating",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coating=SimpleCoating(0.5, 0.5),
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.coating, SimpleCoating)

    def test_create_surface_with_fresnel(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Fresnel",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coating="fresnel",
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.coating, FresnelCoating)

    def test_invalid_z_with_thickness(self, set_test_backend):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="standard",
                comment="Invalid",
                index=1,
                is_stop=False,
                material="air",
                thickness=5,
                z=1,
            )

    def test_absolute_coordinates(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Absolute",
            index=1,
            is_stop=False,
            material="air",
            radius=10,
            conic=0,
            x=1,
            y=2,
            z=3,
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.cs.x == 1
        assert surface.geometry.cs.y == 2
        assert surface.geometry.cs.z == 3
        assert self.factory.use_absolute_cs

    def test_invalid_thickness_for_abs_cs(self, set_test_backend):
        self.factory.use_absolute_cs = True
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="standard",
                comment="Invalid",
                index=1,
                is_stop=False,
                material="air",
                radius=10,
                conic=0,
                thickness=5,
            )

    def test_create_paraxial_standard(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="paraxial",
            comment="Paraxial",
            f=100,
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
        )
        assert isinstance(surface, ParaxialSurface)
        assert surface.f == 100

    def test_invalid_paraxial_surface(self, set_test_backend):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="paraxial",
                comment="Paraxial",
                f=100,
                index=0,
                is_stop=False,
                material="air",
                thickness=5,
            )
