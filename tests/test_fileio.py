import os
import tempfile
from unittest.mock import mock_open, patch

import pytest

from optiland.fileio.optiland_handler import load_obj_from_json, save_obj_to_json
from optiland.fileio.zemax_handler import (
    ZemaxDataModel,
    ZemaxDataParser,
    ZemaxFileSourceHandler,
    load_zemax_file,
)
from optiland.materials import Material
from optiland.optic import Optic
from optiland.samples.objectives import HeliarLens


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
