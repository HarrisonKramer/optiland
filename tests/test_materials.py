from importlib import resources

import optiland.backend as be
import pytest
import numpy as np

from optiland import materials
from .utils import assert_allclose


class TestIdealMaterial:
    def test_ideal_material_n(self, set_test_backend):
        material = materials.IdealMaterial(n=1.5)
        assert material.n(0.5) == 1.5
        assert material.n(1.0) == 1.5
        assert material.n(2.0) == 1.5
        assert material.k(2.0) == 0.0

    def test_ideal_material_k(self, set_test_backend):
        material = materials.IdealMaterial(n=1.5, k=0.2)
        assert material.k(0.5) == 0.2
        assert material.k(1.0) == 0.2
        assert material.k(2.0) == 0.2

    def test_ideal_to_dict(self, set_test_backend):
        material = materials.IdealMaterial(n=1.5, k=0.2)
        assert material.to_dict() == {
            "index": 1.5,
            "absorp": 0.2,
            "type": materials.IdealMaterial.__name__,
        }

    def test_ideal_from_dict(self, set_test_backend):
        material = materials.IdealMaterial.from_dict(
            {"index": 1.5, "absorp": 0.2, "type": materials.IdealMaterial.__name__},
        )
        assert material.n(0.5) == 1.5
        assert material.k(0.5) == 0.2


def test_mirror_material(set_test_backend):
    mirror = materials.Mirror()
    assert mirror.n(0.5) == -1.0
    assert mirror.k(0.5) == 0.0
    assert mirror.n(1.0) == -1.0
    assert mirror.k(1.0) == 0.0


class TestMaterialFile:
    def test_formula_1(self, set_test_backend):
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
            material.n(1.0)

    def test_formula_2(self, set_test_backend):
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
            material.n(1.0)

    def test_formula_3(self, set_test_backend):
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
            material.n(1.0)

    def test_formula_4(self, set_test_backend):
        rel_file = "data-nk/main/CaGdAlO4/Loiko-o.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.9829612788706874)
        assert_allclose(material.n(0.6), 1.9392994674994937)
        assert_allclose(material.n(1.5), 1.9081487808757178)
        assert_allclose(material.abbe(), 40.87771013627357)

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material.n(1.0)

    def test_formula_5(self, set_test_backend):
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
            material.n(1.0)

    def test_formula_6(self, set_test_backend):
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

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material.n(1.0)

    def test_formula_7(self, set_test_backend):
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
            material.n(1.0)

    def test_formula_8(self, set_test_backend):
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

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12]
        with pytest.raises(ValueError):
            material.n(1.0)

    def test_formula_9(self, set_test_backend):
        rel_file = "data-nk/organic/CH4N2O - urea/Rosker-e.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.3), 1.7043928702073146)
        assert_allclose(material.n(0.6), 1.605403788031452)
        assert_allclose(material.n(1.0), 1.5908956870937045)
        assert_allclose(material.abbe(), 34.60221948120884)

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # force invalid coefficients to test the exception
        material.coefficients = [1.0, 0.58, 0.12, 0.87]
        with pytest.raises(ValueError):
            material.n(1.0)

    def test_tabulated_n(self, set_test_backend):
        rel_file = "data-nk/main/Y3Al5O12/Bond.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(1.0), 1.8197)
        assert_allclose(material.n(2.0), 1.8035)
        assert_allclose(material.n(3.0), 1.7855)
        assert_allclose(material.abbe(), 52.043469741225195)

        # This material has no k values, check that it raises a warning
        assert material.k(1.0) == 0.0

        # Test case when no tabulated data available
        material._n = None
        with pytest.raises((ValueError, TypeError)):
            material.n(1.0)

    def test_tabulated_nk(self, set_test_backend):
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
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        with pytest.raises(ValueError):
            material._set_formula_type("formula 2")

    def test_to_dict(self, set_test_backend):
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
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material_dict = {"filename": filename, "type": materials.MaterialFile.__name__}
        assert materials.MaterialFile.from_dict(material_dict).filename == filename


class TestMaterial:
    def test_standard_material(self, set_test_backend):
        material = materials.Material("N-BK7")
        assert material.n(0.5) == 1.5214144757734767
        assert material.k(0.5) == 9.5781e-09
        assert material.abbe() == 64.1673362374998

    def test_nonexistent_material(self, set_test_backend):
        with pytest.raises(ValueError):
            materials.Material("nonexistent material")

        with pytest.raises(ValueError):
            materials.Material(
                "nonexistent material",
                reference="it really does not exist",
            )

    def test_non_robust_failure(self, set_test_backend):
        # There are many materials matches for BK7. Without robust search,
        # this should fail.
        with pytest.raises(ValueError):
            materials.Material("BK7", robust_search=False)

        # There are also many materials matches for BK7 with schott reference.
        with pytest.raises(ValueError):
            materials.Material("BK7", reference="schott", robust_search=False)

    def test_min_wavelength_filtering(self, set_test_backend):
        material = materials.Material("SF11", min_wavelength=2.0)
        df = material._load_dataframe()
        df_filtered = material._find_material_matches(df)

        # Check that all materials have wavelength ranges including 2.0 µm
        assert np.all(df_filtered["max_wavelength"] >= 2.0)
        assert np.all(df_filtered["min_wavelength"] <= 2.0)

    def test_max_wavelength_filtering(self, set_test_backend):
        material = materials.Material("SF11", max_wavelength=2.0)
        df = material._load_dataframe()
        df_filtered = material._find_material_matches(df)

        # Check that all materials have wavelength ranges including 2.0 µm
        assert np.all(df_filtered["max_wavelength"] >= 2.0)
        assert np.all(df_filtered["min_wavelength"] <= 2.0)

    def test_raise_material_error_method(self, set_test_backend):
        material = materials.Material("SF11")
        with pytest.raises(ValueError):
            material._raise_material_error(no_matches=False, multiple_matches=False)

        # Confirm error raise when wavelength ranges are passed
        material = materials.Material("SF11", min_wavelength=0.5, max_wavelength=0.7)
        with pytest.raises(ValueError):
            material._raise_material_error(no_matches=False, multiple_matches=False)

    def test_to_dict(self, set_test_backend):
        material = materials.Material("SF11")
        mat_dict = material.to_dict()
        assert mat_dict == {
            "type": "Material",
            "filename": material.filename,
            "name": "SF11",
            "reference": None,
            "robust_search": True,
            "min_wavelength": None,
            "max_wavelength": None,
        }

    def test_from_dict(self, set_test_backend):
        material_dict = {"name": "SF11", "type": materials.Material.__name__}
        assert materials.Material.from_dict(material_dict).name == "SF11"

    def test_raise_warning(self, set_test_backend):
        materials.Material("LITHOTEC-CAF2")  # prints a warning


@pytest.fixture
def abbe_material():
    return materials.AbbeMaterial(n=1.5, abbe=50)


def test_refractive_index(set_test_backend, abbe_material):
    wavelength = 0.58756  # in microns
    value = abbe_material.n(wavelength)
    assert_allclose(value, 1.4999167964912952)


def test_extinction_coefficient(set_test_backend, abbe_material):
    wavelength = 0.58756  # in microns
    assert abbe_material.k(wavelength) == 0


def test_coefficients(set_test_backend, abbe_material):
    coefficients = abbe_material._get_coefficients()
    assert coefficients.shape == (4,)  # Assuming the polynomial is of degree 3


def test_abbe_to_dict(set_test_backend, abbe_material):
    abbe_dict = abbe_material.to_dict()
    assert abbe_dict == {"type": "AbbeMaterial", "index": 1.5, "abbe": 50}


def test_abbe_from_dict(set_test_backend):
    abbe_dict = {"type": "AbbeMaterial", "index": 1.5, "abbe": 50}
    abbe_material = materials.BaseMaterial.from_dict(abbe_dict)
    assert abbe_material.index == 1.5
    assert abbe_material.abbe == 50


def test_abbe_out_of_bounds_wavelength(set_test_backend):
    abbe_material = materials.AbbeMaterial(n=1.5, abbe=50)
    with pytest.raises(ValueError):
        abbe_material.n(0.3)
    with pytest.raises(ValueError):
        abbe_material.n(0.8)
