import pytest
import numpy as np
from optiland.coordinate_system import CoordinateSystem
from optiland.surfaces.standard_surface import Surface
from optiland.rays import RealRays, ParaxialRays
from optiland.materials import IdealMaterial
from optiland.scatter import LambertianBSDF
from optiland.geometries import Plane
from optiland.coatings import SimpleCoating, FresnelCoating


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
            is_reflective=True
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
