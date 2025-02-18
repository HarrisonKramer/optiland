import os
from unittest.mock import patch, mock_open
import pytest
from optiland.fileio import (
    ZemaxFileReader,
    load_optiland_file,
    save_optiland_file)
from optiland.fileio.optiland_handler import (
    load_obj_from_json,
    save_obj_to_json)
from optiland.optic import Optic
from optiland.materials import Material
from optiland.samples.objectives import HeliarLens
import tempfile


@pytest.fixture
def zemax_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, 'zemax_files/lens1.zmx')
    return filename


@pytest.fixture(scope="module", params=['zemax_files/lens1.zmx', 'zemax_files/lens_thorlabs_iso_8859_1.zmx'])
def zemax_file_formats(request):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, request.param)
    return filename


@pytest.fixture
def zemax_file_reader(zemax_file):
    return ZemaxFileReader(zemax_file)


@pytest.fixture
def zemax_file_formats_reader(zemax_file_formats):
    return ZemaxFileReader(zemax_file_formats)


def replace_line_in_zmx(zmx_file, line_prefix, new_line):
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp.zmx')

    with open(zmx_file, 'r') as file:
        lines = file.readlines()

    with open(temp_file, 'w') as file:
        for line in lines:
            if line.startswith(line_prefix):
                file.write(new_line + '\n')
            else:
                file.write(line)

    return temp_file


class TestZemaxFileReader:
    def test_is_url(self, zemax_file_reader):
        assert zemax_file_reader._is_url('http://www.google.com') is True

    def test_is_not_url(self, zemax_file_reader):
        assert zemax_file_reader._is_url('www.google.com') is False
        assert zemax_file_reader._is_url('I am not a url.') is False

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('tempfile.NamedTemporaryFile')
    def test_configure_source_input_url(self, mock_tempfile, mock_open,
                                        mock_requests_get, zemax_file_reader):
        mock_response = mock_requests_get.return_value
        mock_response.status_code = 200
        mock_response.content = b'Test content'

        temp_file = mock_tempfile.return_value.__enter__.return_value
        temp_file.name = 'temp.zmx'

        filename = 'http://example.com/test.zmx'
        zemax_file_reader.source = filename
        zemax_file_reader._configure_source_input()

        mock_requests_get.assert_called_once_with(filename)
        temp_file.write.assert_called_once_with(b'Test content')
        assert zemax_file_reader.filename == 'temp.zmx'

    @patch('builtins.open', new_callable=mock_open)
    def test_configure_source_input_file(self, mock_open, zemax_file_formats_reader):
        zemax_file_formats_reader.source = 'local_file.zmx'
        zemax_file_formats_reader._configure_source_input()

        assert zemax_file_formats_reader.filename == 'local_file.zmx'
        mock_open.assert_not_called()

    @patch('requests.get')
    def test_configure_source_input_url_download_fail(self, mock_requests_get,
                                                      zemax_file_reader):
        mock_response = mock_requests_get.return_value
        mock_response.status_code = 404

        filename = 'http://example.com/test.zmx'
        zemax_file_reader.source = filename
        with pytest.raises(ValueError, match='Failed to download Zemax file.'):
            zemax_file_reader._configure_source_input()

        mock_requests_get.assert_called_once_with(filename)

    def test_read_nonexistent_file(self, zemax_file_reader):
        zemax_file_reader.filename = 'non_existent_file.zmx'
        with pytest.raises(FileNotFoundError):
            zemax_file_reader._read_file()

    @patch('builtins.open', side_effect=UnicodeError)
    def test_read_file_fail(self, mock_open, zemax_file_reader):
        with pytest.raises(ValueError, match='Failed to read Zemax file.'):
            zemax_file_reader._read_file()

    def test_read_fno(self, zemax_file_reader):
        zemax_file_reader._read_fno(['FNO', '1.5', '0'])
        assert zemax_file_reader.data['aperture']['imageFNO'] == 1.5

    def test_read_paraxial_fno(self, zemax_file_reader):
        zemax_file_reader._read_fno(['FNO', '1.5', '1'])
        assert zemax_file_reader.data['aperture']['paraxialImageFNO'] == 1.5

    def test_read_epd(self, zemax_file_reader):
        zemax_file_reader._read_epd(['ENPD', '2.5'])
        assert zemax_file_reader.data['aperture']['EPD'] == 2.5

    def test_read_object_na(self, zemax_file_reader):
        zemax_file_reader._read_object_na(['OBNA', '0.1', '0'])
        assert zemax_file_reader.data['aperture']['objectNA'] == 0.1

    def test_read_obj_cone_angle(self, zemax_file_reader):
        zemax_file_reader._read_object_na(['OBNA', '0.1', '1'])
        assert zemax_file_reader.data['aperture']['object_cone_angle'] == 0.1

    def test_read_floating_stop(self, zemax_file_reader):
        zemax_file_reader._read_floating_stop(['FLOA', '1'])
        assert zemax_file_reader.data['aperture']['floating_stop']

    def test_read_field_type_angle(self, zemax_file_reader):
        data = ['FTYP', '0', '0', '0', '0', '0', '0', '0']
        zemax_file_reader._read_config_data(data)
        assert zemax_file_reader.data['fields']['type'] == 'angle'

    def test_read_field_type_obj_height(self, zemax_file_reader):
        data = ['FTYP', '1', '0', '0', '0', '0', '0', '0']
        zemax_file_reader._read_config_data(data)
        assert zemax_file_reader.data['fields']['type'] == 'object_height'

    def test_read_field_type_parax_img_height(self, zemax_file_reader):
        data = ['FTYP', '2', '0', '0', '0', '0', '0', '0']
        zemax_file_reader._read_config_data(data)
        assert zemax_file_reader.data['fields']['type'] == \
            'paraxial_image_height'

    def test_read_field_type_real_img_height(self, zemax_file_reader):
        data = ['FTYP', '3', '0', '0', '0', '0', '0', '0']
        zemax_file_reader._read_config_data(data)
        assert zemax_file_reader.data['fields']['type'] == 'real_image_height'

    def test_read_field_type_theodolite(self, zemax_file_reader):
        data = ['FTYP', '4', '0', '0', '0', '0', '0', '0']
        zemax_file_reader._read_config_data(data)
        assert zemax_file_reader.data['fields']['type'] == 'theodolite_angle'

    def test_read_field_type_unsupported(self, zemax_file_reader):
        data = ['FTYP', '10', '0', '0', '0', '0', '0', '0']
        zemax_file_reader._read_config_data(data)
        assert zemax_file_reader.data['fields']['type'] == 'unsupported'

    def test_obj_space_telecentric(self, zemax_file_reader):
        data = ['FTYP', '0', '1', '0', '0', '0', '0', '0']
        zemax_file_reader._read_config_data(data)
        assert zemax_file_reader.data['fields']['object_space_telecentric']

    def test_afocal_image_space(self, zemax_file_reader):
        data = ['FTYP', '0', '0', '0', '0', '0', '0', '1']
        zemax_file_reader._read_config_data(data)
        assert zemax_file_reader.data['fields']['afocal_image_space']

    def test_read_x_fields(self, zemax_file_reader):
        data = ['XFLD', '0', '1', '2']
        zemax_file_reader._read_x_fields(data)
        assert zemax_file_reader.data['fields']['x'] == [0, 1, 2]

    def test_read_y_fields(self, zemax_file_reader):
        data = ['YFLD', '0', '1', '2']
        zemax_file_reader._read_y_fields(data)
        assert zemax_file_reader.data['fields']['y'] == [0, 1, 2]

    def test_read_wavelength(self, zemax_file_reader):
        data = ['WAVM', '1', '0.6']
        zemax_file_reader.data['wavelengths']['data'] = []
        zemax_file_reader._read_wavelength(data)
        assert zemax_file_reader.data['wavelengths']['data'] == [0.6]

    def test_read_conic(self, zemax_file_reader):
        data = ['CONI', '0']
        zemax_file_reader._read_conic(data)
        assert zemax_file_reader._current_surf_data['conic'] == 0

    def test_read_glass(self, zemax_file_reader):
        data = ['GLAS', 'N-BK7', '1', '1', '1', '50']
        zemax_file_reader._read_glass(data)
        mat = zemax_file_reader._current_surf_data['material']
        assert isinstance(mat, Material)

    def test_read_stop(self, zemax_file_reader):
        zemax_file_reader._read_stop([])
        assert zemax_file_reader._current_surf_data['is_stop']

    def test_read_primary_wave(self, zemax_file_reader):
        data = ['PWAV', '1', '0.5']
        zemax_file_reader._read_primary_wave(data)
        assert zemax_file_reader.data['wavelengths']['primary_index'] == 0

    def test_read_mode(self, zemax_file_reader):
        data = ['MODE', 'SEQ']
        zemax_file_reader._read_mode(data)

    def test_read_invalid_mode(self, zemax_file_reader):
        data = ['MODE', 'INVALID']
        with pytest.raises(ValueError):
            zemax_file_reader._read_mode(data)

    def test_read_glass_catalog(self, zemax_file_reader):
        data = ['GCAT', 'SCHOTT', 'Spaghettini']
        zemax_file_reader._read_glass_catalog(data)
        names = ['SCHOTT', 'Spaghettini']
        assert zemax_file_reader.data['glass_catalogs'] == names

    def test_read_surface_standard(self, zemax_file_reader):
        data = ['TYPE', 'STANDARD']
        zemax_file_reader._read_surf_type(data)
        assert zemax_file_reader._current_surf_data['type'] == 'standard'

    def test_read_surface_asphere(self, zemax_file_reader):
        data = ['TYPE', 'EVENASPH']
        zemax_file_reader._read_surf_type(data)
        assert zemax_file_reader._current_surf_data['type'] == 'even_asphere'

    def test_read_surface_invalid(self, zemax_file_reader):
        data = ['TYPE', 'INVALID']
        zemax_file_reader._read_surf_type(data)
        assert zemax_file_reader._current_surf_data['type'] == 'unsupported'

    def test_read_surf_param(self, zemax_file_reader):
        data = ['PARM', '1', '2']
        zemax_file_reader._read_surface_parameter(data)
        val = zemax_file_reader._current_surf_data['param_0']
        assert val == 2

    def test_read_vignette_decenter_x(self, zemax_file_reader):
        data = ['VDXN', '1', '2']
        zemax_file_reader._read_vignette_decenter_x(data)
        val = zemax_file_reader.data['fields']['vignette_decenter_x']
        assert val == [1, 2]

    def test_read_vignette_decenter_y(self, zemax_file_reader):
        data = ['VDYN', '1', '2']
        zemax_file_reader._read_vignette_decenter_y(data)
        val = zemax_file_reader.data['fields']['vignette_decenter_y']
        assert val == [1, 2]

    def test_read_vignette_compress_x(self, zemax_file_reader):
        data = ['VCXN', '1', '2']
        zemax_file_reader._read_vignette_compress_x(data)
        val = zemax_file_reader.data['fields']['vignette_compress_x']
        assert val == [1, 2]

    def test_read_vignette_compress_y(self, zemax_file_reader):
        data = ['VCYN', '1', '2']
        zemax_file_reader._read_vignette_compress_y(data)
        val = zemax_file_reader.data['fields']['vignette_compress_y']
        assert val == [1, 2]

    def test_read_vignette_tangent_angle(self, zemax_file_reader):
        data = ['VANN', '1', '2']
        zemax_file_reader._read_vignette_tangent_angle(data)
        val = zemax_file_reader.data['fields']['vignette_tangent_angle']
        assert val == [1, 2]


class TestZemaxToOpticConverter:
    def test_generate_lens(self, zemax_file_reader):
        lens = zemax_file_reader.generate_lens()
        assert lens is not None
        assert isinstance(lens, Optic)

    def test_generate_asphere(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, 'zemax_files/lens2.zmx')
        zemax_file_reader = ZemaxFileReader(filename)
        lens = zemax_file_reader.generate_lens()
        assert lens is not None
        assert isinstance(lens, Optic)

    def test_generate_lens_with_invalid_mode(self, zemax_file_reader):
        zemax_file_reader.data['surfaces'][0]['type'] = 'invalid'
        with pytest.raises(ValueError, match='Unsupported surface type.'):
            zemax_file_reader.generate_lens()

    def test_configure_fields(self, zemax_file_reader):
        vig = [0.5, 0.5, 0.5]
        zemax_file_reader.data['fields']['vignette_compress_x'] = vig
        zemax_file_reader.data['fields']['vignette_compress_y'] = vig
        lens = zemax_file_reader.generate_lens()
        assert lens.fields.get_vig_factor(Hx=0, Hy=1) == (0.5, 0.5)

    def test_configure_fields_decenter(self, zemax_file_reader):
        zemax_file_reader.data['fields']['vignette_decenter_x'] = [0.5]
        zemax_file_reader.data['fields']['vignette_decenter_y'] = [0.5]
        zemax_file_reader.generate_lens()


def test_save_load_json_obj():
    mat = Material('SF11')
    with tempfile.NamedTemporaryFile(delete=False,
                                     mode='w',
                                     suffix='.json') as temp_file:
        save_obj_to_json(mat, temp_file.name)
    assert os.path.exists(temp_file.name)

    mat2 = load_obj_from_json(Material, temp_file.name)

    assert mat.to_dict() == mat2.to_dict()


def test_load_invalid_json():
    with pytest.raises(FileNotFoundError):
        load_obj_from_json(Material, 'non_existent_file.json')


def test_save_load_optiland_file():
    lens = HeliarLens()
    with tempfile.NamedTemporaryFile(delete=False,
                                     mode='w',
                                     suffix='.json') as temp_file:
        save_optiland_file(lens, temp_file.name)

    lens2 = load_optiland_file(temp_file.name)
    assert lens.to_dict() == lens2.to_dict()
