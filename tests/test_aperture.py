import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland import aperture
from .utils import assert_allclose


@pytest.mark.parametrize(
    "ap_type, value",
    [
        ("EPD", 10),
        ("imageFNO", 3.2),
        ("objectNA", 0.265),
        ("float_by_stop_size", 1.142857),
    ],
)
def test_aperture_generate(set_test_backend, ap_type, value):
    """Check instantiation of aperture"""
    ap = aperture.Aperture(ap_type, value)
    assert ap.value == value


def test_confirm_invalid_ap_type(set_test_backend):
    """Confirm invalid ap_type raises error"""
    with pytest.raises(ValueError):
        aperture.Aperture("invalid_type", 5.0)


def test_obj_space_telecentric(set_test_backend):
    """Confirm error raised when EPD specified with telecentric lens"""
    with pytest.raises(ValueError):
        aperture.Aperture("EPD", 5.0, object_space_telecentric=True)


def test_to_dict(set_test_backend):
    """Check to_dict method"""
    ap = aperture.Aperture("EPD", 10)
    assert ap.to_dict() == {
        "type": "EPD",
        "value": 10,
        "object_space_telecentric": False,
    }


def test_from_dict(set_test_backend):
    """Check from_dict method"""
    ap = aperture.Aperture("EPD", 10)
    ap_dict = ap.to_dict()
    ap2 = aperture.Aperture.from_dict(ap_dict)
    assert ap2.to_dict() == ap.to_dict()


def test_invalid_from_dict(set_test_backend):
    """Check from_dict method with invalid dict"""
    with pytest.raises(ValueError):
        aperture.Aperture.from_dict(
            {"invalid": "I am invalid, unfortunately", "value": 5.0},
        )


def test_confirm_stop_size_floating_stop(set_test_backend):
    lens = Optic()

    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(index=1, radius=22.01359, thickness=3.25896, material="SK16")
    lens.add_surface(index=2, radius=-435.76044, thickness=30, is_stop=True)
    lens.add_surface(index=3)

    stop_diam = 8.4
    stop_idx = 2
    lens.set_aperture(aperture_type="float_by_stop_size", value=stop_diam)

    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_field(y=14)
    lens.add_field(y=20)

    lens.add_wavelength(value=0.48)
    lens.add_wavelength(value=0.55, is_primary=True)
    lens.add_wavelength(value=0.65)

    lens.paraxial.trace(Hy=0, Py=1, wavelength=0.55)
    assert_allclose(lens.surface_group.y[stop_idx], stop_diam / 2)
