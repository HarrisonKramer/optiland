import os
import tempfile
from unittest.mock import mock_open, patch

import pytest

from optiland.fileio.optiland_handler import load_obj_from_json, save_obj_to_json
from optiland.fileio import save_optiland_file, load_optiland_file
from optiland.fileio.zemax_handler import (
    ZemaxDataModel,
    ZemaxDataParser,
    ZemaxFileSourceHandler,
    load_zemax_file,
)
from optiland.materials import Material
from optiland.optic import Optic
from optiland.samples.objectives import HeliarLens
import optiland.backend as be
from .utils import assert_allclose


@pytest.fixture
def zemax_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, "zemax_files/lens1.zmx")
    return filename


@pytest.fixture(
    scope="module",
    params=["zemax_files/lens1.zmx", "zemax_files/lens_thorlabs_iso_8859_1.zmx"],
)
def zemax_file_formats(request):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, request.param)
    return filename


class TestZemaxFileSourceHandler:
    def test_is_url(self):
        handler = ZemaxFileSourceHandler("http://example.com/test.zmx")
        assert handler._is_url()

    def test_is_not_url(self):
        handler = ZemaxFileSourceHandler("not_a_url")
        assert not handler._is_url()

    @patch("requests.get")
    @patch("builtins.open", new_callable=mock_open)
    @patch("tempfile.NamedTemporaryFile")
    def test_get_local_file_url(
        self, mock_tempfile, mock_open, mock_requests_get, zemax_file
    ):
        mock_response = mock_requests_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"Test content"

        temp_file = mock_tempfile.return_value.__enter__.return_value
        temp_file.name = "temp.zmx"

        handler = ZemaxFileSourceHandler("http://example.com/test.zmx")
        local = handler.get_local_file()

        mock_requests_get.assert_called_once_with(
            "http://example.com/test.zmx", timeout=10
        )
        temp_file.write.assert_called_once_with(b"Test content")
        assert local == "temp.zmx"

    @patch("requests.get")
    def test_get_local_file_url_fail(self, mock_requests_get):
        mock_response = mock_requests_get.return_value
        mock_response.status_code = 404

        handler = ZemaxFileSourceHandler("http://example.com/test.zmx")
        with pytest.raises(ValueError, match="Failed to download Zemax file."):
            handler.get_local_file()

    def test_get_local_file_local_path(self, zemax_file):
        handler = ZemaxFileSourceHandler(zemax_file)
        local = handler.get_local_file()
        assert local == zemax_file


class TestZemaxDataParser:
    def setup_method(self):
        self.parser = ZemaxDataParser("dummy")

    def test_read_fno(self):
        self.parser._read_fno(["FNO", "1.5", "0"])
        assert self.parser.data_model.aperture["imageFNO"] == 1.5

    def test_read_epd(self):
        self.parser._read_epd(["ENPD", "2.5"])
        assert self.parser.data_model.aperture["EPD"] == 2.5

    def test_read_object_na(self):
        self.parser._read_object_na(["OBNA", "0.1", "0"])
        assert self.parser.data_model.aperture["objectNA"] == 0.1

    def test_read_conic(self):
        self.parser._read_conic(["CONI", "0"])
        assert self.parser._current_surf_data["conic"] == 0.0

    def test_read_glass(self):
        self.parser._read_glass(["GLAS", "N-BK7", "0", "0", "1.5", "50"])
        mat = self.parser._current_surf_data["material"]
        assert isinstance(mat, Material)

    def test_read_stop(self):
        self.parser._read_stop([])
        assert self.parser._current_surf_data["is_stop"]

    def test_read_mode_valid(self):
        self.parser._read_mode(["MODE", "SEQ"])

    def test_read_mode_invalid(self):
        with pytest.raises(ValueError):
            self.parser._read_mode(["MODE", "NONSEQ"])

    def test_read_surface_type(self):
        self.parser._read_surf_type(["TYPE", "STANDARD"])
        assert self.parser._current_surf_data["type"] == "standard"


class TestEndToEnd:
    def test_load_zemax_file(self, zemax_file):
        optic = load_zemax_file(zemax_file)
        assert isinstance(optic, Optic)

    def test_load_and_convert_asphere(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, "zemax_files/lens2.zmx")
        optic = load_zemax_file(filename)
        assert isinstance(optic, Optic)


def test_save_load_json_obj():
    mat = Material("SF11")
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".json"
    ) as temp_file:
        save_obj_to_json(mat, temp_file.name)
    assert os.path.exists(temp_file.name)

    mat2 = load_obj_from_json(Material, temp_file.name)
    assert mat.to_dict() == mat2.to_dict()


def test_load_invalid_json():
    with pytest.raises(FileNotFoundError):
        load_obj_from_json(Material, "non_existent_file.json")


def test_save_load_optiland_file():
    lens = HeliarLens()
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".json"
    ) as temp_file:
        from optiland.fileio import save_optiland_file, load_optiland_file

        save_optiland_file(lens, temp_file.name)
        lens2 = load_optiland_file(temp_file.name)
    assert lens.to_dict() == lens2.to_dict()


def test_load_legacy_optiland_file_with_field_type():
    """Test loading an Optiland file with the legacy `field_type` key."""
    import json
    from optiland.fields import AngleField
    from optiland.fileio import load_optiland_file

    # 1. Create a modern optic and get its dictionary representation
    lens = HeliarLens()
    lens.set_field_type("angle")
    modern_dict = lens.to_dict()

    # 2. Create a legacy dictionary from the modern one
    legacy_dict = lens.to_dict()
    legacy_dict["fields"]["field_type"] = "angle"
    del legacy_dict["fields"]["field_definition"]

    # 3. Save the legacy dictionary to a temporary file
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".json"
    ) as temp_file:
        json.dump(legacy_dict, temp_file)
        filepath = temp_file.name

    # 4. Load the legacy file
    loaded_lens = load_optiland_file(filepath)

    # 5. Assert that the loaded lens is correct
    assert isinstance(loaded_lens.field_definition, AngleField)
    assert modern_dict == loaded_lens.to_dict()

    os.remove(filepath)


def test_remove_surface_after_load(set_test_backend, tmp_path):
    """
    Test that removing a surface after loading from a file works correctly.
    This reproduces a bug where surface thicknesses were not being deserialized,
    leading to incorrect surface positions after removal.
    """
    # 1. Create a lens and save it
    lens = Optic(name="TestLens")
    lens.add_surface(index=0, thickness=be.inf, material="Air")
    lens.add_surface(
        index=1,
        surface_type="standard",
        material="Air",
        thickness=10,
        radius=150,
    )
    lens.add_surface(
        index=2,
        surface_type="standard",
        material="N-BK7",
        thickness=10,
        radius=150,
        is_stop=True,
    )
    lens.add_surface(
        index=3,
        surface_type="standard",
        material="Air",
        thickness=20,
        radius=be.inf,
    )
    lens.add_surface(index=4)
    lens.set_aperture("float_by_stop_size", 25)

    filepath = tmp_path / "lens.json"
    save_optiland_file(lens, filepath)

    # 2. Load the lens from the file
    loaded_lens = load_optiland_file(filepath)

    # 3. Remove the second surface (the air spacer)
    loaded_lens.surface_group.remove_surface(1)

    # 4. Assert that the positions of the remaining surfaces are correct
    # Original surfaces: 0 (obj), 1 (air), 2 (n-bk7), 3 (air), 4 (img)
    # Positions before removal: inf, 0, 10, 20, 40
    # Thicknesses: inf, 10, 10, 20, 0
    # After removing surf 1:
    # New surfaces: 0 (obj), 1 (n-bk7), 2 (air), 3 (img)
    # New surf 1 (orig 2) z -> 0.0. Its thickness is 10.
    # New surf 2 (orig 3) z -> 0.0 + 10 = 10.0. Its thickness is 20.
    # New surf 3 (orig 4) z -> 10.0 + 20 = 30.0. Its thickness is 0.
    positions = loaded_lens.surface_group.positions.flatten()

    # The object surface's position (index 0) is be.inf and not relevant to the bug.
    # We check the positions of the subsequent surfaces.
    expected_positions_after_object = be.array([0.0, 10.0, 30.0])

    assert_allclose(positions[1:], expected_positions_after_object)
