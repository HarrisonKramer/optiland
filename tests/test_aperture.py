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


def test_to_dict():
    """Check to_dict method"""
    ap = aperture.Aperture('EPD', 10)
    assert ap.to_dict() == {'type': 'EPD', 'value': 10,
                            'object_space_telecentric': False}


def test_from_dict():
    """Check from_dict method"""
    ap = aperture.Aperture('EPD', 10)
    ap_dict = ap.to_dict()
    ap2 = aperture.Aperture.from_dict(ap_dict)
    assert ap2.to_dict() == ap.to_dict()


def test_invalid_from_dict():
    """Check from_dict method with invalid dict"""
    with pytest.raises(ValueError):
        aperture.Aperture.from_dict({'invalid': 'I am invalid, unfortunately',
                                     'value': 5.0})
