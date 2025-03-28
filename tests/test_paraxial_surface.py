import numpy as np
import pytest

from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import ParaxialRays, RealRays
from optiland.scatter import LambertianBSDF
from optiland.surfaces.paraxial_surface import ParaxialSurface


class TestParaxialSurface:
    @pytest.fixture(autouse=True)
    def setup(self):
        cs = CoordinateSystem()
        self.focal_length = 42
        self.geometry = Plane(cs)
        self.material_pre = IdealMaterial(1, 0)
        self.material_post = IdealMaterial(1.5, 0)
        self.aperture = None
        self.coating = SimpleCoating(0.5, 0.5)
        self.bsdf = LambertianBSDF()
        self.surface = ParaxialSurface(
            focal_length=self.focal_length,
            geometry=self.geometry,
            material_pre=self.material_pre,
            material_post=self.material_post,
            is_stop=True,
            aperture=self.aperture,
            coating=self.coating,
            bsdf=self.bsdf,
            is_reflective=True,
        )

    def test_init(self):
        assert self.surface.f == self.focal_length
        assert self.geometry == self.surface.geometry
        assert self.material_pre == self.surface.material_pre
        assert self.material_post == self.surface.material_post
        assert self.aperture == self.surface.aperture
        assert self.coating == self.surface.coating
        assert self.bsdf == self.surface.bsdf
        assert self.surface.is_reflective is True
        assert self.surface.surface_type == "paraxial"

    def test_trace_paraxial_rays(self):
        y = np.array([1])
        u = np.array([0])
        z = np.array([-10])
        w = np.array([1])
        rays = ParaxialRays(y, u, z, w)
        traced_rays = self.surface.trace(rays)
        assert isinstance(traced_rays, ParaxialRays)

    def test_trace_real_rays(self):
        x = np.random.rand(10)
        rays = RealRays(x, x, x, x, x, x, x, x)
        traced_rays = self.surface.trace(rays)
        assert isinstance(traced_rays, RealRays)

    def test_paraxial_surface_perfect_imaging(self):
        lens = Optic()

        # add surfaces
        lens.add_surface(index=0, thickness=np.inf)
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
        assert np.allclose(rays.y, 0, atol=1e-10)

        # confirm all points at exact same z
        assert np.allclose(rays.z, 100, atol=1e-10)

    def test_to_dict(self):
        data = self.surface.to_dict()
        assert data["type"] == "ParaxialSurface"
        assert data["focal_length"] == self.focal_length

    def test_from_dict(self):
        data = self.surface.to_dict()
        surface = ParaxialSurface.from_dict(data)
        assert isinstance(surface, ParaxialSurface)
        assert surface.f == self.focal_length
        assert surface.geometry.to_dict() == self.geometry.to_dict()
        assert surface.material_pre.to_dict() == self.material_pre.to_dict()
        assert surface.material_post.to_dict() == self.material_post.to_dict()
        assert surface.is_stop
        assert surface.aperture == self.aperture
        assert surface.coating.to_dict() == self.coating.to_dict()
        assert surface.bsdf.to_dict() == self.bsdf.to_dict()
        assert surface.is_reflective
        assert surface.semi_aperture is None
        assert np.array_equal(surface.y, np.empty(0))
        assert np.array_equal(surface.u, np.empty(0))
        assert np.array_equal(surface.x, np.empty(0))
        assert np.array_equal(surface.z, np.empty(0))
        assert np.array_equal(surface.L, np.empty(0))
        assert np.array_equal(surface.M, np.empty(0))
        assert np.array_equal(surface.N, np.empty(0))
        assert np.array_equal(surface.intensity, np.empty(0))
        assert np.array_equal(surface.aoi, np.empty(0))
        assert np.array_equal(surface.opd, np.empty(0))
