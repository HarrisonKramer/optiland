# tests/coatings/test_coatings.py
"""
Tests for the coating classes in optiland.coatings.
"""
import pytest

from optiland.coatings import SimpleCoating, FresnelCoating
from optiland.materials import IdealMaterial
from ..utils import assert_allclose


class TestSimpleCoating:
    """
    Tests the SimpleCoating class, which represents a coating with constant
    reflectivity and transmissivity.
    """

    def test_get_coeffs(self, set_test_backend):
        """
        Tests that the transmission and reflection coefficients are returned
        correctly.
        """
        coating = SimpleCoating(reflection=0.1, transmission=0.9)
        # For SimpleCoating, wavelength and angle of incidence are ignored.
        t, r = coating.get_coeffs(wavelength=0.55, aoi=0.1)
        assert t == 0.9
        assert r == 0.1

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of a SimpleCoating instance to a dictionary.
        """
        coating = SimpleCoating(reflection=0.1, transmission=0.9)
        d = coating.to_dict()
        assert d["type"] == "SimpleCoating"
        assert d["reflection"] == 0.1
        assert d["transmission"] == 0.9

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of a SimpleCoating instance from a
        dictionary.
        """
        d = {"type": "SimpleCoating", "reflection": 0.1, "transmission": 0.9}
        coating = SimpleCoating.from_dict(d)
        assert coating.reflection == 0.1
        assert coating.transmission == 0.9


class TestFresnelCoating:
    """
    Tests the FresnelCoating class, which calculates reflection and
    transmission based on the Fresnel equations.
    """

    @pytest.fixture
    def setup_materials(self):
        """
        Provides a pair of materials for testing the Fresnel equations.
        """
        n1 = IdealMaterial(n=1.0)
        n2 = IdealMaterial(n=1.5)
        return n1, n2

    def test_get_coeffs_normal_incidence(self, set_test_backend, setup_materials):
        """
        Tests the Fresnel coefficients for normal incidence (AOI = 0).
        """
        n1, n2 = setup_materials
        coating = FresnelCoating(n1, n2)
        t, r = coating.get_coeffs(wavelength=0.55, aoi=0.0)
        # For normal incidence, r = ((n1-n2)/(n1+n2))^2 = ((1-1.5)/(1+1.5))^2 = 0.04
        assert_allclose(r, 0.04)
        # t = 1 - r
        assert_allclose(t, 0.96)

    def test_get_coeffs_brewster_angle(self, set_test_backend, setup_materials):
        """
        Tests that at Brewster's angle, the reflection for p-polarized light
        is zero.
        """
        n1, n2 = setup_materials
        coating = FresnelCoating(n1, n2)
        brewster_angle = coating._calculate_brewster_angle()
        # For p-polarized light at Brewster's angle, reflectivity should be zero.
        # The get_coeffs method returns the average of s and p polarization.
        # We expect r_p = 0.
        ts, rs, tp, rp = coating._get_fresnel_coeffs(brewster_angle)
        assert_allclose(rp, 0.0, atol=1e-9)

    def test_to_dict(self, set_test_backend, setup_materials):
        """
        Tests the serialization of a FresnelCoating instance to a dictionary.
        """
        n1, n2 = setup_materials
        coating = FresnelCoating(n1, n2)
        d = coating.to_dict()
        assert d["type"] == "FresnelCoating"
        assert d["material1"] == n1.to_dict()
        assert d["material2"] == n2.to_dict()

    def test_from_dict(self, set_test_backend, setup_materials):
        """
        Tests the deserialization of a FresnelCoating instance from a
        dictionary.
        """
        n1, n2 = setup_materials
        d = {
            "type": "FresnelCoating",
            "material1": n1.to_dict(),
            "material2": n2.to_dict(),
        }
        coating = FresnelCoating.from_dict(d)
        assert isinstance(coating.material1, IdealMaterial)
        assert isinstance(coating.material2, IdealMaterial)
        assert coating.material1.n(1) == n1.n(1)
        assert coating.material2.n(1) == n2.n(1)