from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import optiland.backend as be
import pytest

from optiland import analysis
from optiland.optic import Optic
from optiland.samples.objectives import CookeTriplet, TripletTelescopeObjective

matplotlib.use("Agg")  # use non-interactive backend for testing


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

    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(index=1, radius=22.01359, thickness=3.25896, material="SK16")
    lens.add_surface(index=2, radius=-435.76044, thickness=6.00755)
    lens.add_surface(
        index=3,
        radius=-22.21328,
        thickness=0.99997,
        material=("F2", "schott"),
    )
    lens.add_surface(index=4, radius=20.29192, thickness=4.75041, is_stop=True)
    lens.add_surface(index=5, radius=79.68360, thickness=2.95208, material="SK16")
    lens.add_surface(index=6, radius=-18.39533, thickness=42.20778)
    lens.add_surface(index=7)

    lens.set_aperture(aperture_type="EPD", value=10)

    lens.set_field_type(field_type="angle")
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

    def test_airy_disc(self, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        airy_radius = spot.airy_disc_x_y(wavelength=cooke_triplet.primary_wavelength)
        airy_radius_x, airy_radius_y = airy_radius

        assert airy_radius_x[0] == pytest.approx(0.0033403700287742426, abs=1e-9)
        assert airy_radius_x[1] == pytest.approx(0.003579789351003376, abs=1e-9)
        assert airy_radius_x[2] == pytest.approx(0.003825501589577882, abs=1e-9)

        assert airy_radius_y[0] == pytest.approx(0.0033403700287742426, abs=1e-9)
        assert airy_radius_y[1] == pytest.approx(0.003430811760325915, abs=1e-9)
        assert airy_radius_y[2] == pytest.approx(0.0035453238661865244, abs=1e-9)

    @patch("matplotlib.pyplot.show")
    def test_view_spot_diagram(self, mock_show, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        spot.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_spot_diagram_larger_fig(self, mock_show, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        spot.view(figsize=(20, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTripetSpotDiagram:
    @patch("matplotlib.pyplot.show")
    def test_view_spot_diagram(self, mock_show, triplet_four_fields):
        spot = analysis.SpotDiagram(triplet_four_fields)
        spot.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_spot_diagram_larger_fig(self, mock_show, triplet_four_fields):
        spot = analysis.SpotDiagram(triplet_four_fields)
        spot.view(figsize=(20, 10))
        mock_show.assert_called_once()
        plt.close()


class TestCookeTripletEncircledEnergy:
    def test_encircled_energy_centroid(self, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        centroid = encircled_energy.centroid()

        # encircled energy calculation includes randomness, so abs. tolerance
        # is set to 1e-3
        assert centroid[0][0] == pytest.approx(-8.207497747771947e-06, abs=1e-3)
        assert centroid[0][1] == pytest.approx(1.989147771098717e-06, abs=1e-3)

        assert centroid[1][0] == pytest.approx(3.069405792964239e-05, abs=1e-3)
        assert centroid[1][1] == pytest.approx(12.421326489507168, abs=1e-3)

        assert centroid[2][0] == pytest.approx(3.1631726815066986e-07, abs=1e-3)
        assert centroid[2][1] == pytest.approx(18.13502264954927, abs=1e-3)

    @patch("matplotlib.pyplot.show")
    def test_view_encircled_energy(self, mock_show, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        encircled_energy.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_encircled_energy_larger_fig(self, mock_show, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        encircled_energy.view(figsize=(20, 10))
        mock_show.assert_called_once()
        plt.close()


class TestCookeTripletRayFan:
    def test_ray_fan(self, cooke_triplet):
        fan = analysis.RayFan(cooke_triplet)

        assert fan.data["Px"][0] == -1
        assert fan.data["Px"][-1] == 1

        assert fan.data["Py"][0] == -1
        assert fan.data["Py"][-1] == 1

        assert fan.data["(0.0, 0.0)"]["0.48"]["x"][0] == pytest.approx(
            0.00238814980958324,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.0)"]["0.48"]["x"][-1] == pytest.approx(
            -0.00238814980958324,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.0)"]["0.48"]["y"][0] == pytest.approx(
            0.00238814980958324,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.0)"]["0.48"]["y"][-1] == pytest.approx(
            -0.00238814980958324,
            abs=1e-9,
        )

        assert fan.data["(0.0, 0.0)"]["0.55"]["x"][0] == pytest.approx(
            0.004195677081323623,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.0)"]["0.55"]["x"][-1] == pytest.approx(
            -0.004195677081323623,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.0)"]["0.55"]["y"][0] == pytest.approx(
            0.004195677081323623,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.0)"]["0.55"]["y"][-1] == pytest.approx(
            -0.004195677081323623,
            abs=1e-9,
        )

        assert fan.data["(0.0, 0.0)"]["0.65"]["x"][0] == pytest.approx(
            -8.284696919602652e-06,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.0)"]["0.65"]["x"][-1] == pytest.approx(
            8.284696919602652e-06,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.0)"]["0.65"]["y"][0] == pytest.approx(
            -8.284696919602652e-06,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.0)"]["0.65"]["y"][-1] == pytest.approx(
            8.284696919602652e-06,
            abs=1e-9,
        )

        assert fan.data["(0.0, 0.7)"]["0.48"]["x"][0] == pytest.approx(
            0.01973142095198721,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.7)"]["0.48"]["x"][-1] == pytest.approx(
            -0.01973142095198721,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.7)"]["0.48"]["y"][0] == pytest.approx(
            -0.023207115035676296,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.7)"]["0.48"]["y"][-1] == pytest.approx(
            0.03928464835618861,
            abs=1e-9,
        )

        assert fan.data["(0.0, 0.7)"]["0.55"]["x"][0] == pytest.approx(
            0.021420191179537973,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.7)"]["0.55"]["x"][-1] == pytest.approx(
            -0.021420191179537973,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.7)"]["0.55"]["y"][0] == pytest.approx(
            -0.024812371459915994,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.7)"]["0.55"]["y"][-1] == pytest.approx(
            0.04075295155640113,
            abs=1e-9,
        )

        assert fan.data["(0.0, 0.7)"]["0.65"]["x"][0] == pytest.approx(
            0.017025487217305013,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.7)"]["0.65"]["x"][-1] == pytest.approx(
            -0.017025487217305013,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.7)"]["0.65"]["y"][0] == pytest.approx(
            -0.03229666187094615,
            abs=1e-9,
        )
        assert fan.data["(0.0, 0.7)"]["0.65"]["y"][-1] == pytest.approx(
            0.047721942006075935,
            abs=1e-9,
        )

        assert fan.data["(0.0, 1.0)"]["0.48"]["x"][0] == pytest.approx(
            0.01563881685548374,
            abs=1e-9,
        )
        assert fan.data["(0.0, 1.0)"]["0.48"]["x"][-1] == pytest.approx(
            -0.01563881685548374,
            abs=1e-9,
        )
        assert fan.data["(0.0, 1.0)"]["0.48"]["y"][0] == pytest.approx(
            -0.0044989771745065354,
            abs=1e-9,
        )
        assert fan.data["(0.0, 1.0)"]["0.48"]["y"][-1] == pytest.approx(
            0.013000385049824814,
            abs=1e-9,
        )

        assert fan.data["(0.0, 1.0)"]["0.55"]["x"][0] == pytest.approx(
            0.016936433773790505,
            abs=1e-9,
        )
        assert fan.data["(0.0, 1.0)"]["0.55"]["x"][-1] == pytest.approx(
            -0.016936433773790505,
            abs=1e-9,
        )
        assert fan.data["(0.0, 1.0)"]["0.55"]["y"][0] == pytest.approx(
            -0.01705141007843025,
            abs=1e-9,
        )
        assert fan.data["(0.0, 1.0)"]["0.55"]["y"][-1] == pytest.approx(
            0.022501847359645666,
            abs=1e-9,
        )

        assert fan.data["(0.0, 1.0)"]["0.65"]["x"][0] == pytest.approx(
            0.01214534602206907,
            abs=1e-9,
        )
        assert fan.data["(0.0, 1.0)"]["0.65"]["x"][-1] == pytest.approx(
            -0.01214534602206907,
            abs=1e-9,
        )
        assert fan.data["(0.0, 1.0)"]["0.65"]["y"][0] == pytest.approx(
            -0.033957537601747134,
            abs=1e-9,
        )
        assert fan.data["(0.0, 1.0)"]["0.65"]["y"][-1] == pytest.approx(
            0.036545592330593735,
            abs=1e-9,
        )

    @patch("matplotlib.pyplot.show")
    def test_view_ray_fan(self, mock_show, cooke_triplet):
        ray_fan = analysis.RayFan(cooke_triplet)
        ray_fan.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_ray_fan_larger_fig(self, mock_show, cooke_triplet):
        ray_fan = analysis.RayFan(cooke_triplet)
        ray_fan.view(figsize=(20, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTelescopeTripletYYbar:
    @patch("matplotlib.pyplot.show")
    def test_view_yybar(self, mock_show, telescope_objective):
        yybar = analysis.YYbar(telescope_objective)
        yybar.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_yybar_larger_fig(self, mock_show, telescope_objective):
        yybar = analysis.YYbar(telescope_objective)
        yybar.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTelescopeTripletDistortion:
    def test_distortion_values(self, telescope_objective):
        dist = analysis.Distortion(telescope_objective)

        assert dist.data[0][0] == pytest.approx(0.0, abs=1e-9)
        assert dist.data[0][-1] == pytest.approx(0.005950509480884957, abs=1e-9)

        assert dist.data[1][0] == pytest.approx(0.0, abs=1e-9)
        assert dist.data[1][-1] == pytest.approx(0.005786305783771451, abs=1e-9)

        assert dist.data[0][0] == pytest.approx(0.0, abs=1e-9)
        assert dist.data[2][-1] == pytest.approx(0.005720392850412076, abs=1e-9)

    def test_f_theta_distortion(self, telescope_objective):
        dist = analysis.Distortion(telescope_objective, distortion_type="f-theta")

        assert dist.data[0][0] == pytest.approx(0.0, abs=1e-9)
        assert dist.data[0][-1] == pytest.approx(0.016106265133212852, abs=1e-9)

        assert dist.data[1][0] == pytest.approx(0.0, abs=1e-9)
        assert dist.data[1][-1] == pytest.approx(0.015942044760968603, abs=1e-9)

        assert dist.data[0][0] == pytest.approx(0.0, abs=1e-9)
        assert dist.data[2][-1] == pytest.approx(0.015876125134060767, abs=1e-9)

    def test_invalid_distortion_type(self, telescope_objective):
        with pytest.raises(ValueError):
            analysis.Distortion(telescope_objective, distortion_type="invalid")

    @patch("matplotlib.pyplot.show")
    def test_view_distortion(self, mock_show, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        dist.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_distortion_larger_fig(self, mock_show, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        dist.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTelescopeTripletGridDistortion:
    def test_grid_distortion_values(self, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        assert dist.data["max_distortion"] == pytest.approx(
            0.005785718069180374,
            abs=1e-9,
        )

        assert dist.data["xr"].shape == (10, 10)
        assert dist.data["yr"].shape == (10, 10)
        assert dist.data["xp"].shape == (10, 10)
        assert dist.data["yp"].shape == (10, 10)

        assert dist.data["xr"][0, 0] == pytest.approx(-1.2342622299776145, abs=1e-9)
        assert dist.data["xr"][4, 6] == pytest.approx(0.41137984374933073, abs=1e-9)

        assert dist.data["yr"][1, 0] == pytest.approx(-0.959951505834632, abs=1e-9)
        assert dist.data["yr"][2, 6] == pytest.approx(-0.6856458243955965, abs=1e-9)

        assert dist.data["xp"][0, 2] == pytest.approx(-0.6856375010477692, abs=1e-9)
        assert dist.data["xp"][4, 4] == pytest.approx(-0.13712543741510327, abs=1e-9)

        assert dist.data["yp"][-1, 0] == pytest.approx(1.2341908231761498, abs=1e-9)
        assert dist.data["yp"][1, 5] == pytest.approx(-0.9599069415493584, abs=1e-9)

    def test_f_theta_distortion(self, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective, distortion_type="f-theta")

        assert dist.data["max_distortion"] == pytest.approx(
            0.010863278146924825,
            abs=1e-9,
        )

        assert dist.data["xr"].shape == (10, 10)
        assert dist.data["yr"].shape == (10, 10)
        assert dist.data["xp"].shape == (10, 10)
        assert dist.data["yp"].shape == (10, 10)

        assert dist.data["xr"][0, 0] == pytest.approx(-1.2342622299776145, abs=1e-9)
        assert dist.data["xr"][4, 6] == pytest.approx(0.41137984374933073, abs=1e-9)

        assert dist.data["yr"][1, 0] == pytest.approx(-0.959951505834632, abs=1e-9)
        assert dist.data["yr"][2, 6] == pytest.approx(-0.6856458243955965, abs=1e-9)

        assert dist.data["xp"][0, 2] == pytest.approx(-0.6856267573347536, abs=1e-9)
        assert dist.data["xp"][4, 4] == pytest.approx(-0.13712535146695065, abs=1e-9)

        assert dist.data["yp"][-1, 0] == pytest.approx(1.2341281632025562, abs=1e-9)
        assert dist.data["yp"][1, 5] == pytest.approx(-0.9598774602686547, abs=1e-9)

    def test_invalid_distortion_type(self, telescope_objective):
        with pytest.raises(ValueError):
            analysis.GridDistortion(telescope_objective, distortion_type="invalid")

    @patch("matplotlib.pyplot.show")
    def test_view_grid_distortion(self, mock_show, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        dist.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_grid_distortion_larger_fig(self, mock_show, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        dist.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTelescopeTripletFieldCurvature:
    def test_field_curvature_init(self, telescope_objective):
        field_curvature = analysis.FieldCurvature(telescope_objective)
        assert field_curvature.optic == telescope_objective
        assert (
            field_curvature.wavelengths
            == telescope_objective.wavelengths.get_wavelengths()
        )
        assert field_curvature.num_points == 128

    def test_field_curvature_init_with_wavelength(self, telescope_objective):
        field_curvature = analysis.FieldCurvature(
            telescope_objective,
            wavelengths=[0.5, 0.6],
        )
        assert field_curvature.optic == telescope_objective
        assert field_curvature.wavelengths == [0.5, 0.6]
        assert field_curvature.num_points == 128

    def test_field_curvature_init_with_num_points(self, telescope_objective):
        num_points = 256
        field_curvature = analysis.FieldCurvature(
            telescope_objective,
            num_points=num_points,
        )
        assert field_curvature.optic == telescope_objective
        assert (
            field_curvature.wavelengths
            == telescope_objective.wavelengths.get_wavelengths()
        )
        assert field_curvature.num_points == num_points

    def test_field_curvature_init_with_all_parameters(self, telescope_objective):
        num_points = 256
        field_curvature = analysis.FieldCurvature(
            telescope_objective,
            wavelengths=[0.55],
            num_points=num_points,
        )
        assert field_curvature.optic == telescope_objective
        assert field_curvature.wavelengths == [0.55]
        assert field_curvature.num_points == num_points

    @patch("matplotlib.pyplot.show")
    def test_field_curvature_view(self, mock_show, telescope_objective):
        field_curvature = analysis.FieldCurvature(telescope_objective)
        field_curvature.view()
        mock_show.assert_called_once()
        plt.close()

    def test_field_curvature_generate_data(self, telescope_objective):
        f = analysis.FieldCurvature(telescope_objective)

        assert f.data[0][0][89] == pytest.approx(-0.0013062169220806206, abs=1e-9)
        assert f.data[0][1][40] == pytest.approx(0.03435268469825703, abs=1e-9)
        assert f.data[0][1][112] == pytest.approx(0.012502083379998098, abs=1e-9)
        assert f.data[0][0][81] == pytest.approx(0.005363808856891348, abs=1e-9)
        assert f.data[0][0][127] == pytest.approx(-0.041553105637156224, abs=1e-9)
        assert f.data[0][0][40] == pytest.approx(0.02969815644838593, abs=1e-9)
        assert f.data[0][0][57] == pytest.approx(0.021608994058848974, abs=1e-9)
        assert f.data[0][1][45] == pytest.approx(0.03350406866891282, abs=1e-9)
        assert f.data[0][1][74] == pytest.approx(0.026613511090172324, abs=1e-9)
        assert f.data[0][1][94] == pytest.approx(0.01990500178194723, abs=1e-9)

        assert f.data[1][1][55] == pytest.approx(-0.004469963728211546, abs=1e-9)
        assert f.data[1][1][19] == pytest.approx(0.0008003571732224457, abs=1e-9)
        assert f.data[1][1][93] == pytest.approx(-0.015595499139883678, abs=1e-9)
        assert f.data[1][0][15] == pytest.approx(0.0004226818372030349, abs=1e-9)
        assert f.data[1][1][50] == pytest.approx(-0.0034313474749693047, abs=1e-9)
        assert f.data[1][1][50] == pytest.approx(-0.0034313474749693047, abs=1e-9)
        assert f.data[1][0][110] == pytest.approx(-0.05718858127937811, abs=1e-9)
        assert f.data[1][0][89] == pytest.approx(-0.036917737894907106, abs=1e-9)
        assert f.data[1][1][75] == pytest.approx(-0.00961346547634129, abs=1e-9)
        assert f.data[1][0][69] == pytest.approx(-0.021587199726177217, abs=1e-9)

        assert f.data[2][1][62] == pytest.approx(0.059485399479466794, abs=1e-9)
        assert f.data[2][0][103] == pytest.approx(0.015768399161337723, abs=1e-9)
        assert f.data[2][0][0] == pytest.approx(0.06707048647659668, abs=1e-9)
        assert f.data[2][1][68] == pytest.approx(0.05794633552031286, abs=1e-9)
        assert f.data[2][0][6] == pytest.approx(0.06689636005684219, abs=1e-9)
        assert f.data[2][1][40] == pytest.approx(0.06391326892594748, abs=1e-9)
        assert f.data[2][0][88] == pytest.approx(0.029620344519446916, abs=1e-9)
        assert f.data[2][0][5] == pytest.approx(0.06694956529269887, abs=1e-9)
        assert f.data[2][1][98] == pytest.approx(0.048120430272662294, abs=1e-9)
        assert f.data[2][0][5] == pytest.approx(0.06694956529269887, abs=1e-9)


class TestSpotVsField:
    def test_rms_spot_size_vs_field_initialization(self, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        assert spot_vs_field.num_fields == 64
        assert be.array_equal(spot_vs_field._field[:, 1], be.linspace(0, 1, 64))

    def test_rms_spot_radius(self, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        spot_size = spot_vs_field._spot_size
        assert spot_size.shape == (
            64,
            len(telescope_objective.wavelengths.get_wavelengths()),
        )

    @patch("matplotlib.pyplot.show")
    def test_view_spot_vs_field(self, mock_show, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        spot_vs_field.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_spot_vs_field_larger_fig(self, mock_show, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        spot_vs_field.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestWavefrontErrorVsField:
    def test_rms_wave_init(self, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        assert wavefront_error_vs_field.num_fields == 32
        assert be.array_equal(
            wavefront_error_vs_field._field[:, 1],
            be.linspace(0, 1, 32),
        )

    def test_rms_wave(self, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        wavefront_error = wavefront_error_vs_field._wavefront_error
        assert wavefront_error.shape == (
            32,
            len(telescope_objective.wavelengths.get_wavelengths()),
        )

    @patch("matplotlib.pyplot.show")
    def test_view_wave(self, mock_show, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        wavefront_error_vs_field.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_wave_larger_fig(self, mock_show, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        wavefront_error_vs_field.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestPupilAberration:
    def test_initialization(self, telescope_objective):
        pupil_ab = analysis.PupilAberration(telescope_objective)
        assert pupil_ab.optic == telescope_objective
        assert pupil_ab.fields == [(0.0, 0.0), (0.0, 0.7), (0.0, 1.0)]
        assert pupil_ab.wavelengths == [0.4861, 0.5876, 0.6563]
        assert pupil_ab.num_points == 257  # num_points is forced to be odd

    def test_generate_data(self, telescope_objective):
        pupil_ab = analysis.PupilAberration(telescope_objective)
        data = pupil_ab._generate_data()
        assert "Px" in data
        assert "Py" in data
        assert "(0.0, 0.0)" in data
        assert "(0.0, 0.7)" in data
        assert "(0.0, 1.0)" in data
        assert "0.4861" in data["(0.0, 0.0)"]
        assert "0.5876" in data["(0.0, 0.0)"]
        assert "0.6563" in data["(0.0, 0.0)"]
        assert "x" in data["(0.0, 0.0)"]["0.4861"]
        assert "y" in data["(0.0, 0.0)"]["0.4861"]

    @patch("matplotlib.pyplot.show")
    def test_view(self, mock_show, telescope_objective):
        pupil_ab = analysis.PupilAberration(telescope_objective)
        pupil_ab.view()
        mock_show.assert_called_once()
        plt.close()
