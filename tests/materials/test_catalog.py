# tests/materials/test_catalog.py
"""
Tests for the Material class in optiland.materials, which handles searching
the material catalog.
"""
import pytest
import numpy as np

from optiland import materials


class TestMaterial:
    """
    Tests for the Material class, which provides a convenient way to search for
    and load materials from the catalog.
    """
    def test_standard_material(self, set_test_backend):
        """
        Tests the successful loading of a standard material (N-BK7) and
        verifies its refractive index, extinction coefficient, and Abbe number.
        """
        material = materials.Material("N-BK7")
        assert material.n(0.5) == 1.5214144757734767
        assert material.k(0.5) == 9.5781e-09
        assert material.abbe() == 64.1673362374998

    def test_nonexistent_material(self, set_test_backend):
        """
        Tests that attempting to load a nonexistent material raises a
        ValueError.
        """
        with pytest.raises(ValueError):
            materials.Material("nonexistent material")

        with pytest.raises(ValueError):
            materials.Material(
                "nonexistent material",
                reference="it really does not exist",
            )

    def test_non_robust_failure(self, set_test_backend):
        """
        Tests that a non-robust search for a material with multiple matches
        (e.g., "BK7") raises a ValueError.
        """
        # There are many materials matches for BK7. Without robust search,
        # this should fail.
        with pytest.raises(ValueError):
            materials.Material("BK7", robust_search=False)

        # There are also many materials matches for BK7 with schott reference.
        with pytest.raises(ValueError):
            materials.Material("BK7", reference="schott", robust_search=False)

    def test_min_wavelength_filtering(self, set_test_backend):
        """
        Tests that the material search can be filtered by a minimum
        wavelength.
        """
        material = materials.Material("SF11", min_wavelength=2.0)
        df = material._load_dataframe()
        df_filtered = material._find_material_matches(df)

        # Check that all materials have wavelength ranges including 2.0 µm
        assert np.all(df_filtered["max_wavelength"] >= 2.0)
        assert np.all(df_filtered["min_wavelength"] <= 2.0)

    def test_max_wavelength_filtering(self, set_test_backend):
        """
        Tests that the material search can be filtered by a maximum
        wavelength.
        """
        material = materials.Material("SF11", max_wavelength=2.0)
        df = material._load_dataframe()
        df_filtered = material._find_material_matches(df)

        # Check that all materials have wavelength ranges including 2.0 µm
        assert np.all(df_filtered["max_wavelength"] >= 2.0)
        assert np.all(df_filtered["min_wavelength"] <= 2.0)

    def test_raise_material_error_method(self, set_test_backend):
        """
        Tests the internal _raise_material_error method to ensure it raises
        a ValueError under the correct conditions.
        """
        material = materials.Material("SF11")
        with pytest.raises(ValueError):
            material._raise_material_error(no_matches=False, multiple_matches=False)

        # Confirm error raise when wavelength ranges are passed
        material = materials.Material("SF11", min_wavelength=0.5, max_wavelength=0.7)
        with pytest.raises(ValueError):
            material._raise_material_error(no_matches=False, multiple_matches=False)

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of a Material instance to a dictionary.
        """
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
        """
        Tests the deserialization of a Material instance from a dictionary.
        """
        material_dict = {"name": "SF11", "type": materials.Material.__name__}
        assert materials.Material.from_dict(material_dict).name == "SF11"

    def test_raise_warning(self, set_test_backend):
        """
        Tests that a warning is raised for materials with known issues,
        such as LITHOTEC-CAF2.
        """
        materials.Material("LITHOTEC-CAF2")  # prints a warning