import optiland.backend as be
import pytest

from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.materials import IdealMaterial
from optiland.samples.objectives import TessarLens
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.paraxial_surface import ParaxialSurface
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces import SurfaceFactory


class TestSurfaceFactory:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.lens = TessarLens()
        self.surface_group = self.lens.surface_group
        self.factory = SurfaceFactory(self.surface_group)

    def test_create_surface_standard(self):
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

    def test_create_surface_even_asphere(self):
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

    def test_create_surface_odd_asphere(self):
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

    def test_create_surface_polynomial(self):
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

    def test_create_surface_chebyshev(self):
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

    def test_create_surface_object(self):
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

    def test_invalid_surface_type(self):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="invalid",
                comment="Invalid",
                index=1,
                is_stop=False,
                material="air",
                thickness=5,
            )

    def test_invalid_surface_index(self):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="standard",
                comment="Invalid",
                index=42,
                is_stop=False,
                material="air",
                thickness=5,
            )

    def test_create_surface_with_coating(self):
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

    def test_create_surface_with_fresnel(self):
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

    def test_invalid_z_with_thickness(self):
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

    def test_absolute_coordinates(self):
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

    def test_invalid_thickness_for_abs_cs(self):
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

    def test_create_paraxial_standard(self):
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

    def test_invalid_paraxial_surface(self):
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
