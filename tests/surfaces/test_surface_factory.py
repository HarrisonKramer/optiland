# tests/surfaces/test_surface_factory.py
"""
Tests for the SurfaceFactory class in optiland.surfaces.
"""
import optiland.backend as be
import pytest

from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.materials import IdealMaterial
from optiland.samples.objectives import TessarLens
from optiland.interactions.diffractive_model import DiffractiveInteractionModel
from optiland.surfaces.object_surface import ObjectSurface
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces import SurfaceFactory


class TestSurfaceFactory:
    """
    Tests the SurfaceFactory, which is responsible for creating different
    types of surfaces within an optical system.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Sets up a TessarLens and a SurfaceFactory instance for each test.
        """
        self.lens = TessarLens()
        self.surface_group = self.lens.surface_group
        self.factory = SurfaceFactory(self.surface_group)

    def test_create_surface_standard(self, set_test_backend):
        """
        Tests the creation of a standard surface.
        """
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
        """
        Tests the creation of an even aspheric surface.
        """
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
        assert surface.geometry.coefficients == [1, 2, 3]
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_odd_asphere(self, set_test_backend):
        """
        Tests the creation of an odd aspheric surface.
        """
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
        assert surface.geometry.coefficients == [1, 2, 3]
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_polynomial(self, set_test_backend):
        """
        Tests the creation of a polynomial surface.
        """
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
        assert be.array_equal(surface.geometry.coefficients, be.array([[1, 2, 3]]))
        assert surface.geometry.tol == 1e-6
        assert surface.geometry.max_iter == 100
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_chebyshev(self, set_test_backend):
        """
        Tests the creation of a Chebyshev polynomial surface.
        """
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
        assert be.all(surface.geometry.coefficients == be.arange(9).reshape(3, 3))
        assert surface.geometry.tol == 1e-6
        assert surface.geometry.max_iter == 100
        assert surface.geometry.norm_x == 1
        assert surface.geometry.norm_y == 1
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_object(self, set_test_backend):
        """
        Tests that the factory correctly creates an ObjectSurface when the
        surface index is 0.
        """
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
        """
        Tests that providing an unknown surface type raises a ValueError.
        """
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="invalid",
                comment="Invalid",
                index=1,
                is_stop=False,
                material="air",
                thickness=5,
            )

    def test_invalid_surface_index(self, set_test_backend):
        """
        Tests that providing an out-of-bounds surface index raises an
        IndexError.
        """
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
        """
        Tests that a surface can be created with a specified coating.
        """
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
        assert isinstance(surface.interaction_model.coating, SimpleCoating)

    def test_create_surface_with_fresnel(self, set_test_backend):
        """
        Tests that a surface can be created with a Fresnel coating.
        """
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
        assert isinstance(surface.interaction_model.coating, FresnelCoating)

    def test_invalid_z_with_thickness(self, set_test_backend):
        """
        Tests that providing both 'z' (absolute position) and 'thickness'
        (relative position) raises a ValueError.
        """
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
        """
        Tests the creation of a surface using absolute coordinates (x, y, z).
        """
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
        """
        Tests that providing a thickness when the factory is in absolute
        coordinate mode raises a ValueError.
        """
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
        """
        Tests the creation of a paraxial (thin lens) surface.
        """
        surface = self.factory.create_surface(
            surface_type="paraxial",
            comment="Paraxial",
            f=100,
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.interaction_model, ThinLensInteractionModel)
        assert surface.interaction_model.f == 100

    def test_create_grating_surface(self, set_test_backend):
        """
        Tests the creation of a grating surface.
        """
        surface = self.factory.create_surface(
            surface_type="grating",
            comment="Grating",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            grating_period=1,
            grating_order=1,
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.interaction_model, DiffractiveInteractionModel)

    def test_invalid_paraxial_surface(self, set_test_backend):
        """
        Tests that attempting to create a paraxial surface at index 0
        (the object surface) raises a ValueError.
        """
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