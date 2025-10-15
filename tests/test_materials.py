from importlib import resources
from unittest.mock import MagicMock

import optiland.backend as be
import pytest
import numpy as np

from optiland import materials
from optiland.materials.base import BaseMaterial
from optiland.environment.environment import Environment
from optiland.environment.conditions import EnvironmentalConditions
from optiland.materials.ideal import IdealMaterial

from .utils import assert_allclose


class TestBaseMaterial:
    def test_caching(self, set_test_backend):
        class DummyMaterial(BaseMaterial):
            def _calculate_absolute_n(self, wavelength, **kwargs):
                # This method is mocked, so its implementation doesn't matter
                pass

            def _calculate_k(self, wavelength, **kwargs):
                # k is also cached, so we need a dummy implementation
                return 0.0

        material = DummyMaterial()
        material._calculate_absolute_n = MagicMock(return_value=1.5)

        # Mock the environment to isolate the material's behavior
        from optiland.environment.manager import environment_manager

        original_env = environment_manager.get_environment()
        mock_env = Environment(
            medium=IdealMaterial(n=1.0),
            conditions=EnvironmentalConditions(),
        )
        environment_manager.set_environment(mock_env)

        # Test with scalar value
        result1 = material.n(0.5, temperature=25)
        assert result1 == 1.5
        material._calculate_absolute_n.assert_called_once_with(0.5, temperature=25)

        result2 = material.n(0.5, temperature=25)
        assert result2 == 1.5
        material._calculate_absolute_n.assert_called_once()

        # Test with numpy array
        wavelength_np = be.asarray(np.array([0.5, 0.6]))
        material.n(wavelength_np, temperature=25)
        assert material._calculate_absolute_n.call_count == 2

        material.n(wavelength_np, temperature=25)
        assert material._calculate_absolute_n.call_count == 2

        # Test with torch tensor if backend is torch
        if be.get_backend() == "torch":
            # a torch tensor with same values should be a cache hit
            wavelength_torch = be.asarray(np.array([0.5, 0.6]))
            material.n(wavelength_torch, temperature=25)
            assert material._calculate_absolute_n.call_count == 2

            # a torch tensor with different values should be a cache miss
            wavelength_torch_2 = be.asarray(np.array([0.7, 0.8]))
            material.n(wavelength_torch_2, temperature=25)
            assert material._calculate_absolute_n.call_count == 3

            # and a cache hit
            material.n(wavelength_torch_2, temperature=25)
            assert material._calculate_absolute_n.call_count == 3

        # Restore original environment
        environment_manager.set_environment(original_env)


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
            "n": 1.5,
            "k": 0.2,
            "type": materials.IdealMaterial.__name__,
            "relative_to_environment": False,
            "propagation_model": {"class": "HomogeneousPropagation"},
        }

    def test_ideal_from_dict(self, set_test_backend):
        material = materials.IdealMaterial.from_dict(
            {"n": 1.5, "k": 0.2, "type": materials.IdealMaterial.__name__},
        )
        assert material.n(0.5) == 1.5
        assert material.k(0.5) == 0.2


@pytest.fixture
def vacuum_environment():
    from optiland.environment.manager import environment_manager
    from optiland.materials.ideal import IdealMaterial

    original_env = environment_manager.get_environment()
    vacuum_env = Environment(
        medium=IdealMaterial(n=1.0, k=0.0),
        conditions=EnvironmentalConditions(),
    )
    environment_manager.set_environment(vacuum_env)
    yield
    environment_manager.set_environment(original_env)


@pytest.mark.usefixtures("vacuum_environment")
class TestMaterialFileAbsolute:
    def test_formula_4(self, set_test_backend):
        rel_file = "data-nk/main/CaGdAlO4/Loiko-o.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename, is_relative_to_air=False)
        assert_allclose(material.n(0.4), 1.9829612788706874, atol=1e-5)
        assert_allclose(material.n(0.6), 1.9392994674994937, atol=1e-5)
        assert_allclose(material.n(1.5), 1.9081487808757178, atol=1e-5)
        assert_allclose(material.abbe(), 40.87771013627357, atol=1e-5)

    def test_formula_5(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/YbF3/Amotchkina.yml",
            ),
        )
        material = materials.MaterialFile(filename, is_relative_to_air=False)
        assert_allclose(material.n(0.4), 1.5874342875, atol=1e-5)
        assert_allclose(material.n(1.0), 1.487170596, atol=1e-5)
        assert_allclose(material.n(5.0), 1.4844954023999999, atol=1e-5)

    def test_formula_6(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/CO2/Bideau-Mehu.yml",
            ),
        )
        material = materials.MaterialFile(filename, is_relative_to_air=False)
        assert_allclose(material.n(0.4), 1.0004592281255849, atol=1e-5)
        assert_allclose(material.n(1.0), 1.0004424189669583, atol=1e-5)
        assert_allclose(material.n(1.5), 1.0004386003514163, atol=1e-5)

    def test_formula_7(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/Y2O3/Nigara.yml",
            ),
        )
        material = materials.MaterialFile(filename, is_relative_to_air=False)
        material._n_formula = "formula 7"
        material.coefficients = [1.0, 0.58, 0.12, 0.87, 0.21, 0.81]
        assert_allclose(material.n(0.4), 2.0272326, atol=1e-5)
        assert_allclose(material.n(1.0), 3.6137209774932684, atol=1e-5)
        assert_allclose(material.n(1.5), 13.95738334, atol=1e-5)

    def test_formula_8(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/main/AgBr/Schroter.yml",
            ),
        )
        material = materials.MaterialFile(filename, is_relative_to_air=False)
        assert_allclose(material.n(0.5), 2.3094520454859557, atol=1e-5)
        assert_allclose(material.n(0.55), 2.275584479878346, atol=1e-5)
        assert_allclose(material.n(0.65), 2.237243954654548, atol=1e-5)

    def test_formula_9(self, set_test_backend):
        rel_file = "data-nk/organic/CH4N2O - urea/Rosker-e.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename, is_relative_to_air=False)
        assert_allclose(material.n(0.3), 1.7043928702073146, atol=1e-5)
        assert_allclose(material.n(0.6), 1.605403788031452, atol=1e-5)
        assert_allclose(material.n(1.0), 1.5908956870937045, atol=1e-5)

    def test_tabulated_n(self, set_test_backend):
        rel_file = "data-nk/main/Y3Al5O12/Bond.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename, is_relative_to_air=False)
        assert_allclose(material.n(1.0), 1.8197, atol=1e-5)
        assert_allclose(material.n(2.0), 1.8035, atol=1e-5)
        assert_allclose(material.n(3.0), 1.7855, atol=1e-5)

    def test_tabulated_nk(self, set_test_backend):
        rel_file = "data-nk/main/B/Fernandez-Perea.yml"
        filename = str(resources.files("optiland.database").joinpath(rel_file))
        material = materials.MaterialFile(filename, is_relative_to_air=False)
        assert_allclose(material.n(0.005), 0.9947266437313135, atol=1e-5)
        assert_allclose(material.n(0.02), 0.9358854820031199, atol=1e-5)
        assert_allclose(material.n(0.15), 1.990336423662574, atol=1e-5)


class TestMaterialFileRelative:
    def test_formula_1(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/ami/AMTIR-3.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(4), 2.6208713861212907, atol=1e-5)
        assert_allclose(material.n(6), 2.6144067565243265, atol=1e-5)
        assert_allclose(material.n(8), 2.6087270552683854, atol=1e-5)

    def test_formula_2(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/schott/BAFN6.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.6111748495969627, atol=1e-5)
        assert_allclose(material.n(0.8), 1.5803913968709888, atol=1e-5)
        assert_allclose(material.n(1.2), 1.573220342181897, atol=1e-5)

    def test_formula_3(self, set_test_backend):
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/hikari/BASF6.yml",
            ),
        )
        material = materials.MaterialFile(filename)
        assert_allclose(material.n(0.4), 1.6970537915318815, atol=1e-5)
        assert_allclose(material.n(0.5), 1.6767571448173404, atol=1e-5)
        assert_allclose(material.n(0.6), 1.666577226760647, atol=1e-5)

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
            "is_relative_to_air": True,
            "propagation_model": {"class": "HomogeneousPropagation"},
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
        assert_allclose(material.n(0.5), 1.5214144757734767, atol=1e-5)
        assert_allclose(material.k(0.5), 9.5781e-09, atol=1e-10)
        assert_allclose(material.abbe(), 64.1673362374998, atol=1e-5)

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
            "is_relative_to_air": True,
            "propagation_model": {"class": "HomogeneousPropagation"},
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
    assert_allclose(value, 1.4999167964912952, atol=1e-4)


def test_extinction_coefficient(set_test_backend, abbe_material):
    wavelength = 0.58756  # in microns
    assert abbe_material.k(wavelength) == 0


def test_coefficients(set_test_backend, abbe_material):
    coefficients = abbe_material._get_coefficients()
    assert coefficients.shape == (4,)  # Assuming the polynomial is of degree 3


def test_abbe_to_dict(set_test_backend, abbe_material):
    abbe_dict = abbe_material.to_dict()
    assert abbe_dict == {
        "type": "AbbeMaterial",
        "n": 1.5,
        "abbe": 50,
    }


def test_abbe_from_dict(set_test_backend):
    abbe_dict = {"type": "AbbeMaterial", "n": 1.5, "abbe": 50}
    abbe_material = materials.BaseMaterial.from_dict(abbe_dict)
    assert abbe_material.n_val == 1.5
    assert abbe_material.abbe_val == 50


def test_abbe_out_of_bounds_wavelength(set_test_backend):
    abbe_material = materials.AbbeMaterial(n=1.5, abbe=50)
    with pytest.raises(ValueError):
        abbe_material.n(0.3)
    with pytest.raises(ValueError):
        abbe_material.n(0.8)


def test_glasses_selection(set_test_backend):
    glasses = materials.glasses_selection(0.3, 2.5, catalogs=["schott"])
    expected_glasses = [
        "FK3",
        "FK5HTi",
        "K10",
        "LITHOTEC-CAF2",
        "N-BAK1",
        "N-BAK2",
        "N-BK10",
        "N-BK7",
        "N-BK7HT",
        "N-BK7HTi",
        "N-FK5",
        "N-FK51",
        "N-FK51A",
        "N-FK58",
        "N-LAK33B",
        "N-LAK34",
        "N-LAK7",
        "N-PK51",
        "N-PK52A",
        "N-PSK3",
        "N-SK11",
        "N-SK5",
        "N-ZK7",
        "N-ZK7A",
        "P-LAK35",
        "P-SK60",
    ]
    assert glasses == expected_glasses


def test_get_nd_vd(set_test_backend):
    assert materials.get_nd_vd(glass="N-BK7") == (1.5168, 64.17)


def test_downsample_glass_map(set_test_backend):
    glass_dict = {g: materials.get_nd_vd(g) for g in ["N-BK7", "FK3", "FK5HTi", "K10"]}
    downsampled_glass_dict = materials.downsample_glass_map(
        glass_dict,
        num_glasses_to_keep=3,
    )
    expected_downsample_glass_dict = {
        "K10": (1.50137, 56.41),
        "N-BK7": (1.5168, 64.17),
        "FK5HTi": (1.48748, 70.47),
    }
    assert downsampled_glass_dict == expected_downsample_glass_dict


def test_find_closest_glass(set_test_backend):
    assert (
        materials.find_closest_glass(
            nd_vd=(1.5168, 64.17), catalog=["N-BK7", "F5", "SF5"]
        )
        == "N-BK7"
    )


def test_plot_nk():
    from matplotlib.figure import Figure

    mat = materials.Material("BK7")
    fig, axes = materials.plot_nk(mat, wavelength_range=(0.1, 15))
    assert fig is not None
    assert isinstance(fig, Figure)
    assert isinstance(axes, tuple)
    assert len(axes) == 2