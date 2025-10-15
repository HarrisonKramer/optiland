
import pytest

from optiland import backend as be
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.surfaces.factories.surface_factory import SurfaceFactory


class MockSurface:
    def __init__(self):
        self.material_post = IdealMaterial(n=1.0)


class MockSurfaceGroup:
    def __init__(self, num_surfaces):
        self.num_surfaces = num_surfaces
        self.surfaces = [MockSurface()]


def test_nurbs_factory(set_test_backend):
    config = {
        "surface_type": "NurbsGeometry",
        "radius": 100,
        "conic": -1,
        "comment": "NURBS test surface",
        "index": 1,
        "is_stop": False,
        "material": "air",
    }
    factory = SurfaceFactory(surface_group=MockSurfaceGroup(2))
    surf = factory.create_surface(**config)
    assert isinstance(surf.geometry, NurbsGeometry)
    assert be.allclose(surf.geometry.radius, 100)
    assert be.allclose(surf.geometry.k, -1)
