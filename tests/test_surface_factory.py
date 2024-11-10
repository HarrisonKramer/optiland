import pytest
import numpy as np
from optiland.samples.objectives import TessarLens
from optiland.surfaces.surface_factory import SurfaceFactory
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.standard_surface import Surface
from optiland.coatings import SimpleCoating, FresnelCoating
from optiland.materials import IdealMaterial


class TestSurfaceFactory:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.lens = TessarLens()
        self.surface_group = self.lens.surface_group
        self.factory = SurfaceFactory(self.surface_group)

    def test_create_surface_standard(self):
        surface = self.factory.create_surface(
            surface_type='standard',
            index=1,
            is_stop=False,
            material='air',
            thickness=5,
            radius=10,
            conic=0
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_even_asphere(self):
        surface = self.factory.create_surface(
            surface_type='even_asphere',
            index=1,
            is_stop=False,
            material='air',
            thickness=5,
            radius=10,
            conic=0,
            coefficients=[1, 2, 3]
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert surface.geometry.c == [1, 2, 3]
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_polynomial(self):
        surface = self.factory.create_surface(
            surface_type='polynomial',
            index=1,
            is_stop=False,
            material='air',
            thickness=5,
            radius=10,
            conic=0,
            coefficients=[1, 2, 3],
            tol=1e-6,
            max_iter=100
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert surface.geometry.c == [1, 2, 3]
        assert surface.geometry.tol == 1e-6
        assert surface.geometry.max_iter == 100
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_chebyshev(self):
        surface = self.factory.create_surface(
            surface_type='chebyshev',
            index=1,
            is_stop=False,
            material='air',
            thickness=5,
            radius=10,
            conic=0,
            coefficients=np.arange(9).reshape(3, 3),
            tol=1e-6,
            max_iter=100,
            norm_x=1,
            norm_y=1
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert np.all(surface.geometry.c == np.arange(9).reshape(3, 3))
        assert surface.geometry.tol == 1e-6
        assert surface.geometry.max_iter == 100
        assert surface.geometry.norm_x == 1
        assert surface.geometry.norm_y == 1
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_object(self):
        surface = self.factory.create_surface(
            surface_type='standard',
            index=0,
            is_stop=False,
            material='air',
            thickness=5,
            radius=10,
            conic=0
        )
        assert isinstance(surface, ObjectSurface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert isinstance(surface.material_post, IdealMaterial)

    def test_invalid_surface_type(self):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type='invalid',
                index=1,
                is_stop=False,
                material='air',
                thickness=5
            )

    def test_invalid_surface_index(self):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type='standard',
                index=42,
                is_stop=False,
                material='air',
                thickness=5
            )

    def test_create_surface_with_coating(self):
        surface = self.factory.create_surface(
            surface_type='standard',
            index=1,
            is_stop=False,
            material='air',
            thickness=5,
            radius=10,
            conic=0,
            coating=SimpleCoating(0.5, 0.5)
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.coating, SimpleCoating)

    def test_create_surface_with_fresnel(self):
        surface = self.factory.create_surface(
            surface_type='standard',
            index=1,
            is_stop=False,
            material='air',
            thickness=5,
            radius=10,
            conic=0,
            coating='fresnel'
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.coating, FresnelCoating)
