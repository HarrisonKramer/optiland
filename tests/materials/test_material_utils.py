# tests/materials/test_utils.py
"""
Tests for utility functions in optiland.materials.
"""
import pytest
import matplotlib.pyplot as plt

from optiland import materials


def test_glasses_selection(set_test_backend):
    """
    Tests the glasses_selection function to ensure it returns the expected
    list of glasses for a given wavelength range and catalog.
    """
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
    """
    Tests the get_nd_vd function to ensure it returns the correct refractive
    index (nd) and Abbe number (Vd) for a given glass.
    """
    assert materials.get_nd_vd(glass="N-BK7") == (1.5168, 64.17)


def test_downsample_glass_map(set_test_backend):
    """
    Tests the downsample_glass_map function, which reduces the number of
    glasses in a map based on a specified number to keep.
    """
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
    """
    Tests the find_closest_glass function to ensure it correctly identifies
    the closest matching glass from a catalog based on refractive index and
    Abbe number.
    """
    assert (
        materials.find_closest_glass(
            nd_vd=(1.5168, 64.17), catalog=["N-BK7", "F5", "SF5"]
        )
        == "N-BK7"
    )


def test_plot_nk():
    """
    Tests the plot_nk function, which generates a plot of the refractive
    index and extinction coefficient for a given material. This test
    verifies that the function returns a valid matplotlib Figure and Axes.
    """
    mat = materials.Material("BK7")
    fig, axes = materials.plot_nk(mat, wavelength_range=(0.1, 15))
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, list)
    assert len(axes) == 2