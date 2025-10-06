# tests/surfaces/test_grating.py
"""
Tests for grating surfaces in optiland.surfaces.
"""
from optiland.optic import Optic
import optiland.backend as be
import pytest
from ..utils import assert_allclose


@pytest.fixture
def flat_transmission_grating():
    """
    Sets up an optical system with a flat transmission grating.

    The system includes:
    - A flat transmission grating as the third surface.
    - An entrance pupil diameter of 15.
    - Three field points (on-axis, y=10, x=10).
    - A single wavelength of 0.587 µm.
    """
    lens = Optic()

    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(index=1, radius=be.inf, thickness=10)
    lens.add_surface(index=2, radius=be.inf, thickness=5, material="N-BK7")
    lens.add_surface(
        index=3,
        radius=be.inf,
        thickness=30,
        surface_type="grating",
        grating_order=-1,
        grating_period=5.0,
        groove_orientation_angle=0.0,
        is_stop=True,
    )
    lens.add_surface(index=4)

    lens.set_aperture(aperture_type="EPD", value=15)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_field(y=10)
    lens.add_field(y=0, x=10)
    lens.add_wavelength(value=0.587, is_primary=True)
    lens.update_paraxial()
    return lens


@pytest.fixture
def curved_transmission_grating():
    """
    Sets up an optical system with a curved transmission grating.

    The system includes:
    - A curved transmission grating as the third surface.
    - An entrance pupil diameter of 15.
    - Three field points (on-axis, y=10, x=10).
    - A single wavelength of 0.587 µm.
    """
    lens = Optic()

    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(index=1, radius=be.inf, thickness=10)
    lens.add_surface(index=2, radius=be.inf, thickness=5, material="N-BK7")
    lens.add_surface(
        index=3,
        radius=50.0,
        thickness=30,
        conic=1.0,
        surface_type="grating",
        grating_order=-1,
        grating_period=5.0,
        groove_orientation_angle=0.0,
        is_stop=True,
    )
    lens.add_surface(index=4)

    lens.set_aperture(aperture_type="EPD", value=15)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_field(y=10)
    lens.add_field(y=0, x=10)
    lens.add_wavelength(value=0.587, is_primary=True)
    lens.update_paraxial()
    return lens


@pytest.fixture
def curved_reflective_grating():
    """
    Sets up an optical system with a curved reflective grating.

    The system includes:
    - A curved reflective grating as the first surface.
    - An entrance pupil diameter of 15.
    - Three field points (on-axis, y=10, x=10).
    - A single wavelength of 0.587 µm.
    """
    lens = Optic()

    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(
        index=1,
        radius=70,
        thickness=-30,
        material="mirror",
        surface_type="grating",
        is_stop=True,
        grating_period=5.0,
        grating_order=1,
        groove_orientation_angle=0.0,
    )
    lens.add_surface(index=2)

    lens.set_aperture(aperture_type="EPD", value=15)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_field(y=10)
    lens.add_field(y=0, x=10)
    lens.add_wavelength(value=0.587, is_primary=True)
    lens.update_paraxial()
    return lens


def test_flat_grating_transmission(set_test_backend, flat_transmission_grating):
    """
    Tests real ray tracing through a flat transmission grating by verifying
    the output direction cosines of several rays against known values.
    """
    lens = flat_transmission_grating
    wv = 0.587

    # Test axial ray, central field
    ray = lens.trace_generic(Hx=0.0, Hy=0.0, Px=0.0, Py=0.0, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.1174, 0.9930847094])

    # Test marginal ray, central field
    ray = lens.trace_generic(Hx=0.0, Hy=0.0, Px=0.0, Py=1.0, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.1174, 0.9930847094])

    # Test a generic off-axis ray
    ray = lens.trace_generic(Hx=0.2, Hy=0.8, Px=-0.15, Py=0.7, wavelength=wv)
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [0.0345602649, 0.0216899611, 0.9991672201]
    )


def test_curved_grating_transmission(set_test_backend, curved_transmission_grating):
    """
    Tests real ray tracing through a curved transmission grating by verifying
    the output direction cosines of several rays against known values.
    """
    lens = curved_transmission_grating
    wv = 0.587

    # Test axial ray, central field
    ray = lens.trace_generic(Hx=0.0, Hy=0.0, Px=0.0, Py=0.0, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.1174, 0.9930847094])

    # Test marginal ray, central field
    ray = lens.trace_generic(Hx=0.0, Hy=0.0, Px=0.0, Py=1.0, wavelength=wv)
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.0379603895, 0.9992792447]
    )

    # Test a generic off-axis ray
    ray = lens.trace_generic(Hx=0.2, Hy=0.8, Px=-0.15, Py=0.7, wavelength=wv)
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [0.0229384233, 0.0764682608, 0.9968081229]
    )


def test_curved_grating_reflection(set_test_backend, curved_reflective_grating):
    """
    Tests real ray tracing through a curved reflective grating by verifying
    the output direction cosines of a generic ray against known values.
    """
    lens = curved_reflective_grating
    wv = 0.587
    ray = lens.trace_generic(Hx=0.2, Hy=0.8, Px=-0.15, Py=0.7, wavelength=wv)
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [-0.0040370331, -0.4006582284, 0.9162186892]
    )


def test_paraxial_flat_grating_transmission(
    set_test_backend, flat_transmission_grating
):
    """
    Tests paraxial ray tracing through a flat transmission grating by
    verifying the final ray height and angle against known values.
    """
    lens = flat_transmission_grating
    wv = 0.587

    # Trace axial ray
    lens.paraxial.trace(Hy=0.0, Py=0.0, wavelength=wv)
    u = lens.surface_group.u[-1].item()
    y = lens.surface_group.y[-1].item()
    assert_allclose([u, y], [0.1174, 3.522])

    # Trace marginal ray
    lens.paraxial.trace(Hy=0.0, Py=1.0, wavelength=wv)
    u = lens.surface_group.u[-1].item()
    y = lens.surface_group.y[-1].item()
    assert_allclose([u, y], [0.1174, 11.022])

    # Trace a generic ray
    lens.paraxial.trace(Hy=0.8, Py=0.8, wavelength=wv)
    u = lens.surface_group.u[-1].item()
    y = lens.surface_group.y[-1].item()
    assert_allclose([u, y], [0.25794083, 13.73822504])