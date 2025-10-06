# tests/materials/test_file.py
"""
Tests for the MaterialFile class in optiland.materials.
"""
from importlib import resources

import pytest

from optiland import materials
from ..utils import assert_allclose


class TestMaterialFile:
    """
    Tests for the MaterialFile class, which handles loading material data from
    YAML files and calculating refractive indices based on dispersion formulas.
    """
    def test_formula_1(self, set_test_backend):
        """
        Tests dispersion formula 1 (Sellmeier 1) with AMTIR-3 glass.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(4), 2.6208713861212907)
        assert_allclose(material.n(6), 2.6144067565243265)
        assert_allclose(material.n(8), 2.6087270552683854)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_2(self, set_test_backend):
        """
        Tests dispersion formula 2 (Sellmeier 2) with BAFN6 glass.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/schott/BAFN6.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.6111748495969627)
        assert_allclose(material.n(0.8), 1.5803913968709888)
        assert_allclose(material.n(1.2), 1.573220342181897)
        assert_allclose(material.k(0.56), 1.3818058823529405e-08)
        assert_allclose(material.k(0.88), 1.18038e-08)
        assert_allclose(material.abbe(), 48.44594399734635)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_3(self, set_test_backend):
        """
        Tests dispersion formula 3 (Sellmeier 3) with BASF6 glass.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/hikari/BASF6.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.6970537915318815)
        assert_allclose(material.n(0.5), 1.6767571448173404)
        assert_allclose(material.n(0.6), 1.666577226760647)
        assert_allclose(material.k(0.4), 3.3537e-07)
        assert_allclose(material.k(0.5), 2.3945e-08)
        assert_allclose(material.k(0.6), 1.4345e-08)
        assert_allclose(material.abbe(), 42.00944974180074)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_4(self, set_test_backend):
        """
        Tests dispersion formula 4 (Sellmeier 4) with CaGdAlO4.
        """
        rel_file = "data-nk/main/CaGdAlO4/Loiko-o.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.9829612788706874)
        assert_allclose(material.n(0.6), 1.9392994674994937)
        assert_allclose(material.n(1.5), 1.9081487808757178)
        assert_allclose(material.abbe(), 40.87771013627357)

        # This material has no k values, check that it returns 0.0
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_5(self, set_test_backend):
        """
        Tests dispersion formula 5 (Sellmeier 5) with YbF3.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/YbF3/Amotchkina.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.5874342875)
        assert_allclose(material.n(1.0), 1.487170596)
        assert_allclose(material.n(5.0), 1.4844954023999999)
        assert_allclose(material.k(10), 0.004800390585878816)
        assert_allclose(material.k(11), 0.016358499999999998)
        assert_allclose(material.k(12), 0.032864500000000005)
        assert_allclose(material.abbe(), 15.36569851094505)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_6(self, set_test_backend):
        """
        Tests dispersion formula 6 (Conrady) with CO2.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/CO2/Bideau-Mehu.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.0004592281255849)
        assert_allclose(material.n(1.0), 1.0004424189669583)
        assert_allclose(material.n(1.5), 1.0004386003514163)
        assert_allclose(material.abbe(), 76.08072467952312)

        # This material has no k values, check that it returns 0.0
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_7(self, set_test_backend):
        """
        Tests dispersion formula 7 (Herzberger) with fake coefficients,
        as no material in the database currently uses it.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/Y2O3/Nigara.yml",
            ),
        )
        # No material in the database currently uses formula 7, so we fake it
        material = materials.MaterialFile(filename)
        material._n_formula = "formula 7"
        material.coefficients = [1.0, 0.58, 0.12, 0.87, 0.21, 0.81]

        # We test only the equations. These values are meaningless.
        assert_allclose(material.n(0.4), 12.428885495537186)
        assert_allclose(material.n(1.0), 3.6137209774932684)
        assert_allclose(material.n(1.5), 13.532362213339358)
        assert_allclose(material.abbe(), 1.0836925045533496)

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_8(self, set_test_backend):
        """
        Tests dispersion formula 8 (Retro) with AgBr.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/AgBr/Schroter.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.5), 2.3094520454859557)
        assert_allclose(material.n(0.55), 2.275584479878346)
        assert_allclose(material.n(0.65), 2.237243954654548)
        assert_allclose(material.abbe(), 14.551572168536392)

        # This material has no k values, check that it returns 0.0
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_formula_9(self, set_test_backend):
        """
        Tests dispersion formula 9 (Exotic) with urea.
        """
        rel_file = "data-nk/organic/CH4N2O - urea/Rosker-e.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.3), 1.7043928702073146)
        assert_allclose(material.n(0.6), 1.605403788031452)
        assert_allclose(material.n(1.0), 1.5908956870937045)
        assert_allclose(material.abbe(), 34.60221948120884)

        # This material has no k values, check that it returns 0.0
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material._calculate_n(1.0)

    def test_tabulated_n(self, set_test_backend):
        """
        Tests tabulated refractive index data with Y3Al5O12.
        """
        rel_file = "data-nk/main/Y3Al5O12/Bond.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(1.0), 1.8197)
        assert_allclose(material.n(2.0), 1.8035)
        assert_allclose(material.n(3.0), 1.7855)
        assert_allclose(material.abbe(), 52.043469741225195)

        # This material has no k values, check that it returns 0.0
        assert material.k(1.0) == 0.0

        # Test case when no tabulated data available
        material._n = None
        with pytest.raises((ValueError, TypeError)):
            material._calculate_n(1.0)

    def test_tabulated_nk(self, set_test_backend):
        """
        Tests tabulated refractive index and extinction coefficient data with B.
        """
        rel_file = "data-nk/main/B/Fernandez-Perea.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.005), 0.9947266437313135)
        assert_allclose(material.n(0.02), 0.9358854820031199)
        assert_allclose(material.n(0.15), 1.990336423662574)
        assert_allclose(material.k(0.005), 0.0038685437228138607)
        assert_allclose(material.k(0.02), 0.008158161793528261)
        assert_allclose(material.k(0.15), 1.7791319513647896)

    def test_set_formula_type_twice(self, set_test_backend):
        """
        Tests that setting the formula type twice raises a ValueError.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        with pytest.raises(ValueError):
            material._set_formula_type("formula 2")

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of a MaterialFile instance to a dictionary.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert material.to_dict() == {
            "filename": filename,
            "type": materials.MaterialFile.__name__,
        }

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of a MaterialFile instance from a dictionary.
        """
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material_dict = {"filename": filename, "type": materials.MaterialFile.__name__}
        assert materials.MaterialFile.from_dict(material_dict).filename == filename