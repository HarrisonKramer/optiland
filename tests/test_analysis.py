import pytest
import numpy as np
from unittest.mock import patch
from optiland.optics import Optic
from optiland import analysis
from optiland.samples.objectives import TripletTelescopeObjective, CookeTriplet


@pytest.fixture
def cooke_triplet():
    return CookeTriplet()


@pytest.fixture
def telescope_objective():
    return TripletTelescopeObjective()


@pytest.fixture
def triplet_four_fields():
    lens = Optic()

    lens.surface_group.surfaces = []

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=22.01359, thickness=3.25896,
                     material='SK16')
    lens.add_surface(index=2, radius=-435.76044, thickness=6.00755)
    lens.add_surface(index=3, radius=-22.21328, thickness=0.99997,
                     material=('F2', 'schott'))
    lens.add_surface(index=4, radius=20.29192, thickness=4.75041,
                     is_stop=True)
    lens.add_surface(index=5, radius=79.68360, thickness=2.95208,
                     material='SK16')
    lens.add_surface(index=6, radius=-18.39533, thickness=42.20778)
    lens.add_surface(index=7)

    lens.set_aperture(aperture_type='EPD', value=10)

    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_field(y=10)
    lens.add_field(y=15)
    lens.add_field(y=20)

    lens.add_wavelength(value=0.48)
    lens.add_wavelength(value=0.55, is_primary=True)
    lens.add_wavelength(value=0.65)

    lens.update_paraxial()
    return lens


class TestCookeTripetSpotDiagram:
    def test_spot_geometric_radius(self, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        geo_radius = spot.geometric_spot_radius()

        assert geo_radius[0][0] == pytest.approx(0.00597244087781, abs=1e-9)
        assert geo_radius[0][1] == pytest.approx(0.00628645771124, abs=1e-9)
        assert geo_radius[0][2] == pytest.approx(0.00931911440064, abs=1e-9)

        assert geo_radius[1][0] == pytest.approx(0.03717783072826, abs=1e-9)
        assert geo_radius[1][1] == pytest.approx(0.03864613392848, abs=1e-9)
        assert geo_radius[1][2] == pytest.approx(0.04561512437816, abs=1e-9)

        assert geo_radius[2][0] == pytest.approx(0.01951655430245, abs=1e-9)
        assert geo_radius[2][1] == pytest.approx(0.02342659090311, abs=1e-9)
        assert geo_radius[2][2] == pytest.approx(0.03747033587405, abs=1e-9)

    def test_spot_rms_radius(self, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        rms_radius = spot.rms_spot_radius()

        assert rms_radius[0][0] == pytest.approx(0.003791335461448, abs=1e-9)
        assert rms_radius[0][1] == pytest.approx(0.004293689564257, abs=1e-9)
        assert rms_radius[0][2] == pytest.approx(0.006195618755672, abs=1e-9)

        assert rms_radius[1][0] == pytest.approx(0.015694600107671, abs=1e-9)
        assert rms_radius[1][1] == pytest.approx(0.016786721284464, abs=1e-9)
        assert rms_radius[1][2] == pytest.approx(0.019109151416248, abs=1e-9)

        assert rms_radius[2][0] == pytest.approx(0.013229165357157, abs=1e-9)
        assert rms_radius[2][1] == pytest.approx(0.012081348897953, abs=1e-9)
        assert rms_radius[2][2] == pytest.approx(0.013596802321537, abs=1e-9)

    @patch('matplotlib.pyplot.show')
    def test_view_spot_diagram(self, mock_show, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        spot.view()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_view_spot_diagram_larger_fig(self, mock_show, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        spot.view(figsize=(20, 10))
        mock_show.assert_called_once()


class TestTripetSpotDiagram:
    @patch('matplotlib.pyplot.show')
    def test_view_spot_diagram(self, mock_show, triplet_four_fields):
        spot = analysis.SpotDiagram(triplet_four_fields)
        spot.view()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_view_spot_diagram_larger_fig(self, mock_show,
                                          triplet_four_fields):
        spot = analysis.SpotDiagram(triplet_four_fields)
        spot.view(figsize=(20, 10))
        mock_show.assert_called_once()


class TestCookeTripletEncircledEnergy:

    def test_encircled_energy_centroid(self, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        centroid = encircled_energy.centroid()

        # encircled energy calculation includes randomness, so abs. tolerance
        # is set to 1e-3
        assert centroid[0][0] == pytest.approx(-8.207497747771947e-06,
                                               abs=1e-3)
        assert centroid[0][1] == pytest.approx(1.989147771098717e-06, abs=1e-3)

        assert centroid[1][0] == pytest.approx(3.069405792964239e-05, abs=1e-3)
        assert centroid[1][1] == pytest.approx(12.421326489507168, abs=1e-3)

        assert centroid[2][0] == pytest.approx(3.1631726815066986e-07,
                                               abs=1e-3)
        assert centroid[2][1] == pytest.approx(18.13502264954927, abs=1e-3)

    @patch('matplotlib.pyplot.show')
    def test_view_encircled_energy(self, mock_show, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        encircled_energy.view()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_view_encircled_energy_larger_fig(self, mock_show, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        encircled_energy.view(figsize=(20, 10))
        mock_show.assert_called_once()


class TestCookeTripletRayFan:
    def test_ray_fan(self, cooke_triplet):
        fan = analysis.RayFan(cooke_triplet)

        assert fan.data['Px'][0] == -1
        assert fan.data['Px'][-1] == 1

        assert fan.data['Py'][0] == -1
        assert fan.data['Py'][-1] == 1

        assert fan.data['(0.0, 0.0)']['0.48']['x'][0] == \
            pytest.approx(0.00230694465588267, abs=1e-9)
        assert fan.data['(0.0, 0.0)']['0.48']['x'][-1] == \
            pytest.approx(-0.0024693549632838, abs=1e-9)
        assert fan.data['(0.0, 0.0)']['0.48']['y'][0] == \
            pytest.approx(0.00230694465588267, abs=1e-9)
        assert fan.data['(0.0, 0.0)']['0.48']['y'][-1] == \
            pytest.approx(-0.0024693549632838, abs=1e-9)

        assert fan.data['(0.0, 0.0)']['0.55']['x'][0] == \
            pytest.approx(0.00411447192762305, abs=1e-9)
        assert fan.data['(0.0, 0.0)']['0.55']['x'][-1] == \
            pytest.approx(-0.0042768822350241, abs=1e-9)
        assert fan.data['(0.0, 0.0)']['0.55']['y'][0] == \
            pytest.approx(0.00411447192762305, abs=1e-9)
        assert fan.data['(0.0, 0.0)']['0.55']['y'][-1] == \
            pytest.approx(-0.0042768822350241, abs=1e-9)

        assert fan.data['(0.0, 0.0)']['0.65']['x'][0] == \
            pytest.approx(-8.948985061928e-05, abs=1e-9)
        assert fan.data['(0.0, 0.0)']['0.65']['x'][-1] == \
            pytest.approx(-7.292045678185e-05, abs=1e-9)
        assert fan.data['(0.0, 0.0)']['0.65']['y'][0] == \
            pytest.approx(-8.948985061928e-05, abs=1e-9)
        assert fan.data['(0.0, 0.0)']['0.65']['y'][-1] == \
            pytest.approx(-7.292045678185e-05, abs=1e-9)

        assert fan.data['(0.0, 0.7)']['0.48']['x'][0] == \
            pytest.approx(0.01976688587767077, abs=1e-9)
        assert fan.data['(0.0, 0.7)']['0.48']['x'][-1] == \
            pytest.approx(-0.0196959560263178, abs=1e-9)
        assert fan.data['(0.0, 0.7)']['0.48']['y'][0] == \
            pytest.approx(-0.0232699816183430, abs=1e-9)
        assert fan.data['(0.0, 0.7)']['0.48']['y'][-1] == \
            pytest.approx(0.03922178177355029, abs=1e-9)

        assert fan.data['(0.0, 0.7)']['0.55']['x'][0] == \
            pytest.approx(0.02145565610521620, abs=1e-9)
        assert fan.data['(0.0, 0.7)']['0.55']['x'][-1] == \
            pytest.approx(-0.0213847262538632, abs=1e-9)
        assert fan.data['(0.0, 0.7)']['0.55']['y'][0] == \
            pytest.approx(-0.0248752380425489, abs=1e-9)
        assert fan.data['(0.0, 0.7)']['0.55']['y'][-1] == \
            pytest.approx(0.04069008497376458, abs=1e-9)

        assert fan.data['(0.0, 0.7)']['0.65']['x'][0] == \
            pytest.approx(0.01706095214297792, abs=1e-9)
        assert fan.data['(0.0, 0.7)']['0.65']['x'][-1] == \
            pytest.approx(-0.0169900222916250, abs=1e-9)
        assert fan.data['(0.0, 0.7)']['0.65']['y'][0] == \
            pytest.approx(-0.0323595284535702, abs=1e-9)
        assert fan.data['(0.0, 0.7)']['0.65']['y'][-1] == \
            pytest.approx(0.04765907542344116, abs=1e-9)

        assert fan.data['(0.0, 1.0)']['0.48']['x'][0] == \
            pytest.approx(0.01571814948112706, abs=1e-9)
        assert fan.data['(0.0, 1.0)']['0.48']['x'][-1] == \
            pytest.approx(-0.0155594842298404, abs=1e-9)
        assert fan.data['(0.0, 1.0)']['0.48']['y'][0] == \
            pytest.approx(-0.0043495236774511, abs=1e-9)
        assert fan.data['(0.0, 1.0)']['0.48']['y'][-1] == \
            pytest.approx(0.01314983854687312, abs=1e-9)

        assert fan.data['(0.0, 1.0)']['0.55']['x'][0] == \
            pytest.approx(0.01701576639943294, abs=1e-9)
        assert fan.data['(0.0, 1.0)']['0.55']['x'][-1] == \
            pytest.approx(-0.0168571011481462, abs=1e-9)
        assert fan.data['(0.0, 1.0)']['0.55']['y'][0] == \
            pytest.approx(-0.0169019565813819, abs=1e-9)
        assert fan.data['(0.0, 1.0)']['0.55']['y'][-1] == \
            pytest.approx(0.02265130085670819, abs=1e-9)

        assert fan.data['(0.0, 1.0)']['0.65']['x'][0] == \
            pytest.approx(0.01222467864771505, abs=1e-9)
        assert fan.data['(0.0, 1.0)']['0.65']['x'][-1] == \
            pytest.approx(-0.0120660133964284, abs=1e-9)
        assert fan.data['(0.0, 1.0)']['0.65']['y'][0] == \
            pytest.approx(-0.0338080841047059, abs=1e-9)
        assert fan.data['(0.0, 1.0)']['0.65']['y'][-1] == \
            pytest.approx(0.03669504582764205, abs=1e-9)

    @patch('matplotlib.pyplot.show')
    def test_view_ray_fan(self, mock_show, cooke_triplet):
        ray_fan = analysis.RayFan(cooke_triplet)
        ray_fan.view()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_view_ray_fan_larger_fig(self, mock_show, cooke_triplet):
        ray_fan = analysis.RayFan(cooke_triplet)
        ray_fan.view(figsize=(20, 10))
        mock_show.assert_called_once()


class TestTelescopeTripletYYbar:
    @patch('matplotlib.pyplot.show')
    def test_view_yybar(self, mock_show, telescope_objective):
        yybar = analysis.YYbar(telescope_objective)
        yybar.view()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_view_yybar_larger_fig(self, mock_show, telescope_objective):
        yybar = analysis.YYbar(telescope_objective)
        yybar.view(figsize=(12.4, 10))
        mock_show.assert_called_once()


class TestTelescopeTripletDistortion:
    def test_distortion_values(self, telescope_objective):
        dist = analysis.Distortion(telescope_objective)

        assert dist.data[0][0] == pytest.approx(0.0, abs=1e-9)
        assert dist.data[0][-1] == \
            pytest.approx(0.005950509480884957, abs=1e-9)

        assert dist.data[1][0] == pytest.approx(0.0, abs=1e-9)
        assert dist.data[1][-1] == \
            pytest.approx(0.005786305783771451, abs=1e-9)

        assert dist.data[0][0] == pytest.approx(0.0, abs=1e-9)
        assert dist.data[2][-1] == \
            pytest.approx(0.005720392850412076, abs=1e-9)

    @patch('matplotlib.pyplot.show')
    def test_view_distortion(self, mock_show, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        dist.view()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_view_distortion_larger_fig(self, mock_show, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        dist.view(figsize=(12.4, 10))
        mock_show.assert_called_once()


class TestTelescopeTripletGridDistortion:
    def test_grid_distortion_values(self, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)

        assert dist.data['max_distortion'] == \
            pytest.approx(0.005785718069180374, abs=1e-9)

        assert dist.data['xr'].shape == (10, 10)
        assert dist.data['yr'].shape == (10, 10)
        assert dist.data['xp'].shape == (10, 10)
        assert dist.data['yp'].shape == (10, 10)

        assert dist.data['xr'][0, 0] == pytest.approx(1.2342622299776145,
                                                      abs=1e-9)
        assert dist.data['xr'][4, 6] == pytest.approx(-0.41137984374933073,
                                                      abs=1e-9)

        assert dist.data['yr'][1, 0] == pytest.approx(-0.959951505834632,
                                                      abs=1e-9)
        assert dist.data['yr'][2, 6] == pytest.approx(-0.6856458243955965,
                                                      abs=1e-9)

        assert dist.data['xp'][0, 2] == pytest.approx(0.6856375010477692,
                                                      abs=1e-9)
        assert dist.data['xp'][4, 4] == pytest.approx(0.13712543741510327,
                                                      abs=1e-9)

        assert dist.data['yp'][-1, 0] == pytest.approx(1.2341908231761498,
                                                       abs=1e-9)
        assert dist.data['yp'][1, 5] == pytest.approx(-0.9599069415493584,
                                                      abs=1e-9)

    @patch('matplotlib.pyplot.show')
    def test_view_grid_distortion(self, mock_show, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        dist.view()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_view_grid_distortion_larger_fig(self, mock_show,
                                             telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        dist.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
