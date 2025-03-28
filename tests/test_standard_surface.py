import numpy as np
import pytest

from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.materials import IdealMaterial
from optiland.rays import ParaxialRays, RealRays
from optiland.scatter import LambertianBSDF
from optiland.surfaces.standard_surface import Surface


class TestSurface:
    @pytest.fixture(autouse=True)
    def setup(self):
        cs = CoordinateSystem()
        self.geometry = Plane(cs)
        self.material_pre = IdealMaterial(1, 0)
        self.material_post = IdealMaterial(1.5, 0)
        self.aperture = None
        self.coating = SimpleCoating(0.5, 0.5)
        self.bsdf = LambertianBSDF()
        self.surface = Surface(
            geometry=self.geometry,
            material_pre=self.material_pre,
            material_post=self.material_post,
            is_stop=True,
            aperture=self.aperture,
            coating=self.coating,
            bsdf=self.bsdf,
            is_reflective=True,
        )

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

    def test_set_semi_aperture(self):
        r_max = 10.0
        self.surface.set_semi_aperture(r_max)
        assert self.surface.semi_aperture == r_max

    def test_reset(self):
        self.surface.reset()
        assert len(self.surface.y) == 0
        assert len(self.surface.u) == 0
        assert len(self.surface.x) == 0
        assert len(self.surface.z) == 0
        assert len(self.surface.L) == 0
        assert len(self.surface.M) == 0
        assert len(self.surface.N) == 0
        assert len(self.surface.intensity) == 0
        assert len(self.surface.aoi) == 0
        assert len(self.surface.opd) == 0

    def test_set_fresnel_coating(self):
        self.surface.set_fresnel_coating()
        assert isinstance(self.surface.coating, FresnelCoating)

    def test_is_rotationally_symmetric(self):
        self.geometry.is_symmetric = True
        self.geometry.cs.rx = 0
        self.geometry.cs.ry = 0
        self.geometry.cs.x = 0
        self.geometry.cs.y = 0
        assert self.surface.is_rotationally_symmetric()

    def test_is_rotationally_symmetric_false(self):
        self.geometry.is_symmetric = False
        assert not self.surface.is_rotationally_symmetric()

        self.geometry.is_symmetric = True
        self.geometry.cs.rx = 0
        self.geometry.cs.ry = 0.1
        self.geometry.cs.x = 0
        self.geometry.cs.y = 0
        assert not self.surface.is_rotationally_symmetric()

    def test_to_dict(self):
        data = self.surface.to_dict()
        assert data["type"] == "Surface"

    def test_from_dict(self):
        data = self.surface.to_dict()
        surface = Surface.from_dict(data)
        assert isinstance(surface, Surface)
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

    def test_from_dict_missing_type(self):
        data = self.surface.to_dict()
        del data["type"]
        with pytest.raises(ValueError):
            Surface.from_dict(data)
