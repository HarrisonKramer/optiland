from unittest.mock import patch, mock_open

import pytest

from optiland.materials import material_utils


class TestMaterialUtils:
    @patch(
        "builtins.open",
        mock_open(
            read_data="""filename,min_wavelength,max_wavelength,group,filename_no_ext
glass/schott/N-BK7.yml,0.3,2.5,glass,N-BK7
glass/schott/SF11.yml,0.4,2.5,glass,SF11
other/not_a_glass.yml,0.1,5.0,other,not_a_glass
glass/invalid/invalid_wavelength.yml,a,b,glass,invalid_wavelength
"""
        ),
    )
    @patch("importlib.resources.files")
    def test_glasses_selection(self, mock_resources, set_test_backend):
        # Based on the mock data, the valid ranges are:
        # N-BK7: [0.3, 2.5]
        # SF11:  [0.4, 2.5]

        # Test basic filtering: requested range is [0.4, 2.4]
        # N-BK7 contains it (0.3<=0.4, 2.5>=2.4)
        # SF11 contains it (0.4<=0.4, 2.5>=2.4)
        glasses = material_utils.glasses_selection(0.4, 2.4, catalogs=["schott"])
        assert sorted(glasses) == ["N-BK7", "SF11"]

        # Test wavelength boundaries
        # Requested range [0.3, 2.5] is only met by N-BK7
        glasses = material_utils.glasses_selection(0.3, 2.5, catalogs=["schott"])
        assert glasses == ["N-BK7"]

        # Requested range [0.4, 2.5] is met by both
        glasses = material_utils.glasses_selection(0.4, 2.5, catalogs=["schott"])
        assert sorted(glasses) == ["N-BK7", "SF11"]

        # Requested range [0.2, 2.6] is met by none
        glasses = material_utils.glasses_selection(0.2, 2.6, catalogs=["schott"])
        assert glasses == []

        # Test catalog filtering
        glasses = material_utils.glasses_selection(0.4, 2.4, catalogs=["other"])
        assert glasses == []

        # Test with no catalog filter
        glasses = material_utils.glasses_selection(0.4, 2.4, catalogs=None)
        assert sorted(glasses) == ["N-BK7", "SF11"]

    @patch("pathlib.Path.open", new_callable=mock_open, read_data="SPECS:\n  nd: 1.5\n  Vd: 60.0")
    @patch("optiland.materials.material_utils.Material")
    def test_get_nd_vd_success(self, mock_material, mock_path_open, set_test_backend):
        # Mock the Material class to return a fake path
        mock_instance = mock_material.return_value
        mock_instance._retrieve_file.return_value = ("mock_path.yml", {})

        nd, vd = material_utils.get_nd_vd("any_glass")
        assert nd == 1.5
        assert vd == 60.0

    @patch("pathlib.Path.open", new_callable=mock_open, read_data="OTHER_DATA:\n  key: value")
    @patch("optiland.materials.material_utils.Material")
    def test_get_nd_vd_missing_specs(self, mock_material, mock_path_open, set_test_backend):
        # Mock the Material class
        mock_instance = mock_material.return_value
        mock_instance._retrieve_file.return_value = ("mock_path.yml", {})

        nd, vd = material_utils.get_nd_vd("any_glass")
        assert nd == 0
        assert vd == 0

    def test_downsample_glass_map(self, set_test_backend):
        glass_dict = {
            "G1": (1.5, 60), "G2": (1.6, 50), "G3": (1.7, 40),
            "G4": (1.51, 61), "G5": (1.61, 51), "G6": (1.71, 41),
        }
        num_to_keep = 3
        downsampled = material_utils.downsample_glass_map(glass_dict, num_to_keep)
        assert len(downsampled) == num_to_keep
        assert set(downsampled.keys()).issubset(glass_dict.keys())

        with pytest.raises(AssertionError):
            material_utils.downsample_glass_map(glass_dict, 10)
        with pytest.raises(AssertionError):
            material_utils.downsample_glass_map(glass_dict, 1)

    def test_downsample_glass_map_warning(self, set_test_backend):
        # This data is guaranteed to only have 2 clusters
        glass_dict = {
            "G1": (1.5, 60), "G2": (1.5, 60), "G3": (1.5, 60),
            "G4": (1.8, 20), "G5": (1.8, 20),
        }
        with pytest.warns(UserWarning, match="K-Means produced only"):
            material_utils.downsample_glass_map(glass_dict, 3)

    @patch("optiland.materials.material_utils.plot_glass_map")
    def test_get_neighbour_glasses(self, mock_plot, set_test_backend):
        glass_dict = {
            "G1": (1.5, 60), "G2": (1.6, 50), "G3": (1.7, 40),
            "G4": (1.51, 61), "G5": (1.8, 30),
        }
        neighbours = material_utils.get_neighbour_glasses(
            "G1", glass_dict=glass_dict, num_neighbours=2
        )
        assert neighbours == ["G4", "G2"]

        material_utils.get_neighbour_glasses(
            "G1", glass_dict=glass_dict, num_neighbours=2, plot=True
        )
        mock_plot.assert_called_once()

    @patch("optiland.materials.material_utils.get_nd_vd")
    def test_find_closest_glass(self, mock_get_nd_vd, set_test_backend):
        glass_dict = {
            "G1": (1.5, 60), "G2": (1.6, 50), "G3": (1.7, 40),
        }
        mock_get_nd_vd.side_effect = lambda g: glass_dict[g]
        closest = material_utils.find_closest_glass(
            (1.51, 61), catalog=["G1", "G2", "G3"]
        )
        assert closest == "G1"

    @patch("matplotlib.pyplot.subplots")
    @patch("optiland.materials.material_utils.get_nd_vd")
    def test_plot_glass_map(self, mock_get_nd_vd, mock_subplots, set_test_backend):
        from unittest.mock import MagicMock
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        glass_dict = {"G1": (1.5, 60), "G2": (1.6, 50)}
        mock_get_nd_vd.side_effect = lambda g: glass_dict.get(g, (0, 0))

        fig, ax = material_utils.plot_glass_map(["G1", "G2"], highlights=["G1"])

        assert fig is mock_fig
        assert ax is mock_ax
        mock_ax.scatter.assert_called()
        mock_ax.text.assert_called()

    @patch("matplotlib.pyplot.subplots")
    def test_plot_nk(self, mock_subplots, set_test_backend):
        from unittest.mock import MagicMock
        import optiland.backend as be
        mock_fig, mock_ax_n = MagicMock(), MagicMock()
        mock_ax_k = MagicMock()
        mock_ax_n.twinx.return_value = mock_ax_k
        mock_ax_n.get_legend_handles_labels.return_value = ([], [])
        mock_ax_k.get_legend_handles_labels.return_value = ([], [])
        mock_subplots.return_value = (mock_fig, mock_ax_n)

        mock_material = MagicMock()
        mock_material.material_data = {
            "min_wavelength": 0.3, "max_wavelength": 2.5,
            "category_name_full": "Test", "reference": "TEST"
        }
        mock_material.n.return_value = be.ones(10)
        mock_material.k.return_value = be.zeros(10)

        material_utils.plot_nk(mock_material)
        mock_ax_n.plot.assert_called()
        mock_ax_k.plot.assert_called()

        with pytest.warns(UserWarning):
            material_utils.plot_nk(mock_material, wavelength_range=(0.1, 3.0))

        with pytest.raises(ValueError):
            material_utils.plot_nk(mock_material, wavelength_range=(0.1, 0.2, 0.3))

        mock_material.material_data = {}
        with pytest.raises(ValueError):
            material_utils.plot_nk(mock_material)
