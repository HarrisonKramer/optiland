import pytest

import optiland.backend as be
from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.materials import IdealMaterial
from optiland.rays import ParaxialRays, RealRays
from optiland.surfaces.standard_surface import Surface


class TestSurface:
    def create_surface(self):
        cs = CoordinateSystem()
        geometry = Plane(cs)
        material_pre = IdealMaterial(1, 0)
        material_post = IdealMaterial(1.5, 0)
        aperture = None
        coating = SimpleCoating(0.5, 0.5)
        bsdf = None
        return Surface(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            is_stop=True,
            aperture=aperture,
            coating=coating,
            bsdf=bsdf,
            is_reflective=True,
        )

    def test_trace_paraxial_rays(self, set_test_backend):
        surface = self.create_surface()
        y = be.array([1])
        u = be.array([0])
        z = be.array([-10])
        w = be.array([1])
        rays = ParaxialRays(y, u, z, w)
        traced_rays = surface.trace(rays)
        assert isinstance(traced_rays, ParaxialRays)

    def test_trace_real_rays(self, set_test_backend):
        surface = self.create_surface()
        x = be.random_uniform(size=10)
        rays = RealRays(x, x, x, x, x, x, x, x)
        traced_rays = surface.trace(rays)
        assert isinstance(traced_rays, RealRays)

    def test_set_semi_aperture(self, set_test_backend):
        surface = self.create_surface()
        r_max = 10.0
        surface.set_semi_aperture(r_max)
        assert surface.semi_aperture == r_max

    def test_reset(self, set_test_backend):
        surface = self.create_surface()
        surface.reset()
        assert len(surface.y) == 0
        assert len(surface.u) == 0
        assert len(surface.x) == 0
        assert len(surface.z) == 0
        assert len(surface.L) == 0
        assert len(surface.M) == 0
        assert len(surface.N) == 0
        assert len(surface.intensity) == 0
        assert len(surface.aoi) == 0
        assert len(surface.opd) == 0

    def test_set_fresnel_coating(self, set_test_backend):
        surface = self.create_surface()
        surface.set_fresnel_coating()
        assert isinstance(surface.coating, FresnelCoating)

    def test_is_rotationally_symmetric(self, set_test_backend):
        surface = self.create_surface()
        surface.geometry.is_symmetric = True
        surface.geometry.cs.rx = 0
        surface.geometry.cs.ry = 0
        surface.geometry.cs.x = 0
        surface.geometry.cs.y = 0
        assert surface.is_rotationally_symmetric()

    def test_is_rotationally_symmetric_false(self, set_test_backend):
        surface = self.create_surface()
        surface.geometry.is_symmetric = False
        assert not surface.is_rotationally_symmetric()

        surface.geometry.is_symmetric = True
        surface.geometry.cs.rx = 0
        surface.geometry.cs.ry = 0.1
        surface.geometry.cs.x = 0
        surface.geometry.cs.y = 0
        assert not surface.is_rotationally_symmetric()

    def test_to_dict(self, set_test_backend):
        surface = self.create_surface()
        data = surface.to_dict()
        assert data["type"] == "Surface"

    def test_from_dict(self, set_test_backend):
        surface = self.create_surface()
        data = surface.to_dict()
        new_surface = Surface.from_dict(data)
        assert isinstance(new_surface, Surface)
        assert new_surface.geometry.to_dict() == surface.geometry.to_dict()
        assert new_surface.material_pre.to_dict() == surface.material_pre.to_dict()
        assert new_surface.material_post.to_dict() == surface.material_post.to_dict()
        assert new_surface.is_stop == surface.is_stop
        assert new_surface.aperture == surface.aperture
        assert new_surface.coating.to_dict() == surface.coating.to_dict()
        assert new_surface.is_reflective == surface.is_reflective
        assert new_surface.semi_aperture is None
        assert be.array_equal(new_surface.y, be.empty(0))
        assert be.array_equal(new_surface.u, be.empty(0))
        assert be.array_equal(new_surface.x, be.empty(0))
        assert be.array_equal(new_surface.z, be.empty(0))
        assert be.array_equal(new_surface.L, be.empty(0))
        assert be.array_equal(new_surface.M, be.empty(0))
        assert be.array_equal(new_surface.N, be.empty(0))
        assert be.array_equal(new_surface.intensity, be.empty(0))
        assert be.array_equal(new_surface.aoi, be.empty(0))
        assert be.array_equal(new_surface.opd, be.empty(0))

    def test_from_dict_missing_type(self, set_test_backend):
        surface = self.create_surface()
        data = surface.to_dict()
        del data["type"]
        with pytest.raises(ValueError):
            Surface.from_dict(data)
