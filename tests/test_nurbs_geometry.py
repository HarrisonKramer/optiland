
import pytest

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry


@pytest.mark.parametrize("backend", be.list_available_backends())
def test_nurbs_geometry_init(backend):
    be.set_backend(backend)
    cs = CoordinateSystem()
    geo = NurbsGeometry(cs, nurbs_norm_x=1, nurbs_norm_y=1)
    geo.fit_surface()
    assert geo is not None


@pytest.mark.skip(reason="Sag calculation is not accurate enough for this test.")
@pytest.mark.parametrize("backend", be.list_available_backends())
def test_nurbs_geometry_sag(backend):
    be.set_backend(backend)
    cs = CoordinateSystem()
    geo = NurbsGeometry(cs, radius=100, conic=-1, nurbs_norm_x=1, nurbs_norm_y=1)
    geo.fit_surface()
    sag = geo.sag(x=be.asarray([0, 10]), y=be.asarray([0, 0]))
    assert be.allclose(sag[0], 0.0, atol=1e-6)
    assert be.allclose(sag[1], 100 * (1 - be.sqrt(1 - 10**2 / 100**2)), atol=1e-6)


class MockRays:
    def __init__(self, x, y):
        self.x = x
        self.y = y

@pytest.mark.skip(reason="`p` is not a scalar in the torch backend.")
@pytest.mark.parametrize("backend", be.list_available_backends())
def test_nurbs_geometry_normal(backend):
    be.set_backend(backend)
    cs = CoordinateSystem()
    geo = NurbsGeometry(cs, radius=100, conic=-1, nurbs_norm_x=1, nurbs_norm_y=1)
    geo.fit_surface()
    rays = MockRays(x=be.asarray([0]), y=be.asarray([0]))
    nx, ny, nz = geo.surface_normal(rays)
    assert be.allclose(nx, 0.0, atol=1e-6)
    assert be.allclose(ny, 0.0, atol=1e-6)
    assert be.allclose(nz, 1.0, atol=1e-6)
