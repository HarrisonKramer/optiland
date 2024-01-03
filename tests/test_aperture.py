import pytest
from optiland import aperture


@pytest.mark.parametrize('ap_type, value', [('EPD', 10), ('imageFNO', 3.2),
                                            ('objectNA', 0.265)])
def test_aperture_generate(ap_type, value):
    """Check instantiation of aperture"""
    ap = aperture.Aperture(ap_type, value)
    assert ap.value == value


@pytest.mark.parametrize('ap_type, value', [('EPd', 5.0), ('FNO', 6.2),
                                            ('imageNA', 0.01)])
def test_confirm_nonvalid_input(ap_type, value):
    """Confirm nonvalid inputs raise error"""
    with pytest.raises(ValueError):
        aperture.Aperture(ap_type, value)
