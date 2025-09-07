import optiland.backend as be
import pytest

from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import ParaxialRays, RealRays
from optiland.surfaces.paraxial_surface import ParaxialSurface
from .utils import assert_allclose


@pytest.fixture
def surface(set_test_backend):
    cs = CoordinateSystem()
    focal_length = 42
    geometry = Plane(cs)
    material_pre = IdealMaterial(1, 0)
    material_post = IdealMaterial(1.5, 0)
    aperture = None
    coating = SimpleCoating(0.5, 0.5)
    bsdf = None
    return ParaxialSurface(
        focal_length=focal_length,
        geometry=geometry,
        material_pre=material_pre,
        material_post=material_post,
        is_stop=True,
        aperture=aperture,
        coating=coating,
        bsdf=bsdf,
        is_reflective=True,
    )


class TestParaxialSurface:
    def test_init(self, surface):
        assert surface.f == be.array(42)
        assert surface.geometry is not None
        assert surface.material_pre is not None
        assert surface.material_post is not None
        assert surface.aperture is None
        assert surface.coating is not None
        assert surface.bsdf is None
        assert surface.is_reflective is True
        assert surface.surface_type == "paraxial"

    def test_trace_paraxial_rays(self, surface):
        y = be.array([1])
        u = be.array([0])
        z = be.array([-10])
        w = be.array([1])
        rays = ParaxialRays(y, u, z, w)
        traced_rays = surface.trace(rays)
        assert isinstance(traced_rays, ParaxialRays)

    def test_trace_real_rays(self, surface):
        x = be.random_uniform(size=10)
        rays = RealRays(x, x, x, x, x, x, x, x)
        traced_rays = surface.trace(rays)
        assert isinstance(traced_rays, RealRays)

    def test_paraxial_surface_perfect_imaging(self, set_test_backend):
        lens = Optic()

        # add surfaces
        lens.add_surface(index=0, thickness=be.inf)
        lens.add_surface(
            index=1,
            surface_type="paraxial",
            thickness=100,
            f=100,
            is_stop=True,
        )
        lens.add_surface(index=2)

        # add aperture
        lens.set_aperture(aperture_type="EPD", value=20)

        # add field
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        # lens.add_field(y=5)

        # add wavelength
        lens.add_wavelength(value=0.55, is_primary=True)

        rays = lens.trace(
            Hx=0,
            Hy=0,
            wavelength=0.55,
            distribution="uniform",
            num_rays=32,
        )

        # confirm all points exactly on axis
        assert_allclose(rays.y, 0)

        # confirm all points at exact same z
        assert_allclose(rays.z, 100)

    def test_to_dict(self, surface):
        data = surface.to_dict()
        assert data["type"] == "ParaxialSurface"
        assert data["focal_length"] == 42

    def test_from_dict(self, surface):
        data = surface.to_dict()
        new_surface = ParaxialSurface.from_dict(data)
        assert isinstance(new_surface, ParaxialSurface)
        assert new_surface.f == 42
        assert new_surface.geometry.to_dict() == surface.geometry.to_dict()
        assert new_surface.material_pre.to_dict() == surface.material_pre.to_dict()
        assert new_surface.material_post.to_dict() == surface.material_post.to_dict()
        assert new_surface.is_stop
        assert new_surface.aperture == surface.aperture
        assert new_surface.coating.to_dict() == surface.coating.to_dict()
        assert new_surface.is_reflective
        assert new_surface.semi_aperture is None
        assert_allclose(new_surface.y, be.empty(0))
        assert_allclose(new_surface.u, be.empty(0))
        assert_allclose(new_surface.x, be.empty(0))
        assert_allclose(new_surface.z, be.empty(0))
        assert_allclose(new_surface.L, be.empty(0))
        assert_allclose(new_surface.M, be.empty(0))
        assert_allclose(new_surface.N, be.empty(0))
        assert_allclose(new_surface.intensity, be.empty(0))
        assert_allclose(new_surface.aoi, be.empty(0))
        assert_allclose(new_surface.opd, be.empty(0))
