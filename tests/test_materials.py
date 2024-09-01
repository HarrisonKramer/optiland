import pytest
from optiland.materials import IdealMaterial


class TestIdealMaterial:
    def test_ideal_material_n(self):
        material = IdealMaterial(n=1.5)
        assert material.n(0.5) == 1.5
        assert material.n(1.0) == 1.5
        assert material.n(2.0) == 1.5
        assert material.k(2.0) == 0.0

    def test_ideal_material_k(self):
        material = IdealMaterial(n=1.5, k=0.2)
        assert material.k(0.5) == 0.2
        assert material.k(1.0) == 0.2
        assert material.k(2.0) == 0.2

    def test_ideal_material_abbe(self):
        material = IdealMaterial(n=1.5)
        assert material.abbe() == pytest.approx(0.0, abs=1e-10)

    def test_ideal_material_abbe_with_k(self):
        material = IdealMaterial(n=1.5, k=0.2)
        assert material.abbe() == pytest.approx(0.0, abs=1e-10)
