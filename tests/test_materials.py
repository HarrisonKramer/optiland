import os
import pytest
from optiland import materials


class TestIdealMaterial:
    def test_ideal_material_n(self):
        material = materials.IdealMaterial(n=1.5)
        assert material.n(0.5) == 1.5
        assert material.n(1.0) == 1.5
        assert material.n(2.0) == 1.5
        assert material.k(2.0) == 0.0

    def test_ideal_material_k(self):
        material = materials.IdealMaterial(n=1.5, k=0.2)
        assert material.k(0.5) == 0.2
        assert material.k(1.0) == 0.2
        assert material.k(2.0) == 0.2

    def test_ideal_material_abbe(self):
        material = materials.IdealMaterial(n=1.5)
        assert material.abbe() == pytest.approx(0.0, abs=1e-10)

    def test_ideal_material_abbe_with_k(self):
        material = materials.IdealMaterial(n=1.5, k=0.2)
        assert material.abbe() == pytest.approx(0.0, abs=1e-10)


def test_mirror_material():
    mirror = materials.Mirror()
    assert mirror.n(0.5) == -1.0
    assert mirror.k(0.5) == 0.0
    assert mirror.n(1.0) == -1.0
    assert mirror.k(1.0) == 0.0


class TestMaterialFile:
    def test_formula_1(self):
        filename = os.path.join(os.path.dirname(__file__),
                                '../database/data-nk/glass/ami/AMTIR-3.yml')
        material = materials.MaterialFile(filename)
        assert material.n(4) == pytest.approx(2.6208713861212907, abs=1e-10)
        assert material.n(6) == pytest.approx(2.6144067565243265, abs=1e-10)
        assert material.n(8) == pytest.approx(2.6087270552683854, abs=1e-10)

    def test_formula_2(self):
        filename = os.path.join(os.path.dirname(__file__),
                                '../database/data-nk/glass/schott/BAFN6.yml')
        material = materials.MaterialFile(filename)
        assert material.n(0.4) == pytest.approx(1.6111748495969627, abs=1e-10)
        assert material.n(0.8) == pytest.approx(1.5803913968709888, abs=1e-10)
        assert material.n(1.2) == pytest.approx(1.573220342181897, abs=1e-10)
        assert material.k(0.56) == pytest.approx(1.3818058823529405e-08,
                                                 abs=1e-10)
        assert material.k(0.88) == pytest.approx(1.18038e-08, abs=1e-10)
        assert material.abbe() == pytest.approx(48.44594399734635, abs=1e-10)

    def test_formula_3(self):
        filename = os.path.join(os.path.dirname(__file__),
                                '../database/data-nk/glass/hikari/BASF6.yml')
        material = materials.MaterialFile(filename)
        assert material.n(0.4) == pytest.approx(1.6970537915318815, abs=1e-10)
        assert material.n(0.5) == pytest.approx(1.6767571448173404, abs=1e-10)
        assert material.n(0.6) == pytest.approx(1.666577226760647, abs=1e-10)
        assert material.k(0.4) == pytest.approx(3.3537e-07, abs=1e-10)
        assert material.k(0.5) == pytest.approx(2.3945e-08, abs=1e-10)
        assert material.k(0.6) == pytest.approx(1.4345e-08, abs=1e-10)
        assert material.abbe() == pytest.approx(42.00944974180074, abs=1e-10)
