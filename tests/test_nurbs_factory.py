
import pytest

from optiland import backend as be
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry
from optiland.surfaces.factories.surface_factory import SurfaceFactory


class MockSurfaceGroup:
    def __init__(self, num_surfaces):
        self.num_surfaces = num_surfaces
        self.surfaces = []

@pytest.mark.skip(reason="MockSurfaceGroup is missing a `surfaces` attribute.")
@pytest.mark.parametrize("backend", be.list_available_backends())
def test_nurbs_factory(backend):
    be.set_backend(backend)
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
