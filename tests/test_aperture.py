import pytest
from optiland import aperture


@pytest.mark.parametrize('ap_type, value', [('EPD', 10), ('imageFNO', 3.2),
                                            ('objectNA', 0.265)])
def test_aperture_generate(ap_type, value):
    """Check instantiation of aperture"""
    ap = aperture.Aperture(ap_type, value)
    assert ap.value == value


def test_confirm_invalid_ap_type():
    """Confirm invalid ap_type raises error"""
    with pytest.raises(ValueError):
        aperture.Aperture('invalid_type', 5.0)


def test_obj_space_telecentric():
    """Confirm error raised when EPD specified with telecentric lens"""
    with pytest.raises(ValueError):
        aperture.Aperture('EPD', 5.0, object_space_telecentric=True)
