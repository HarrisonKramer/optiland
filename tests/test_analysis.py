from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optiland.backend as be
import pytest

from optiland import analysis
from optiland.optic import Optic
from optiland.samples.objectives import CookeTriplet, TripletTelescopeObjective
from optiland.physical_apertures import RectangularAperture
from optiland.rays import RealRays
from .utils import assert_allclose

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
    def test_spot_geometric_radius(self, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        geo_radius = spot.geometric_spot_radius()
        
        assert_allclose(geo_radius[0][0], 0.00597244087781)
        assert_allclose(geo_radius[0][1], 0.00628645771124)
        assert_allclose(geo_radius[0][2], 0.00931911440064)

        assert_allclose(geo_radius[1][0], 0.03717783072826)
        assert_allclose(geo_radius[1][1], 0.03864613392848)
        assert_allclose(geo_radius[1][2], 0.04561512437816)

        assert_allclose(geo_radius[2][0], 0.01951655430245)
        assert_allclose(geo_radius[2][1], 0.02342659090311)
        assert_allclose(geo_radius[2][2], 0.03747033587405)

    def test_spot_rms_radius(self, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        rms_radius = spot.rms_spot_radius()

        assert_allclose(rms_radius[0][0], 0.003791335461448)
        assert_allclose(rms_radius[0][1], 0.004293689564257)
        assert_allclose(rms_radius[0][2], 0.006195618755672)

        assert_allclose(rms_radius[1][0], 0.015694600107671)
        assert_allclose(rms_radius[1][1], 0.016786721284464)
        assert_allclose(rms_radius[1][2], 0.019109151416248)

        assert_allclose(rms_radius[2][0], 0.013229165357157)
        assert_allclose(rms_radius[2][1], 0.012081348897953)
        assert_allclose(rms_radius[2][2], 0.013596802321537)

    def test_airy_disc(self, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        airy_radius = spot.airy_disc_x_y(wavelength=cooke_triplet.primary_wavelength)
        airy_radius_x, airy_radius_y = airy_radius

        assert_allclose(airy_radius_x[0], 0.0033403700287742426)
        assert_allclose(airy_radius_x[1], 0.003579789351003376)
        assert_allclose(airy_radius_x[2], 0.003825501589577882)
        
        assert_allclose(airy_radius_y[0], 0.0033403700287742426)
        assert_allclose(airy_radius_y[1], 0.003430811760325915)
        assert_allclose(airy_radius_y[2], 0.0035453238661865244)
    
    @patch("matplotlib.pyplot.show")
    def test_airy_disc_in_view_spot_diagram(self, mock_show, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        spot.view(add_airy_disk=True)
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_spot_diagram(self, mock_show, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        spot.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_spot_diagram_larger_fig(self, mock_show, set_test_backend, cooke_triplet):
        spot = analysis.SpotDiagram(cooke_triplet)
        spot.view(figsize=(20, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTripletSpotDiagram:
    @patch("matplotlib.pyplot.show")
    def test_view_spot_diagram(self, mock_show, set_test_backend, triplet_four_fields):
        spot = analysis.SpotDiagram(triplet_four_fields)
        spot.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_spot_diagram_larger_fig(self, mock_show, set_test_backend, triplet_four_fields):
        spot = analysis.SpotDiagram(triplet_four_fields)
        spot.view(figsize=(20, 10))
        mock_show.assert_called_once()
        plt.close()


class TestCookeTripletEncircledEnergy:
    def test_encircled_energy_centroid(self, set_test_backend, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        centroid = encircled_energy.centroid()

        # encircled energy calculation includes randomness, so abs. tolerance
        # is set to 1e-3
        
        assert_allclose(centroid[0][0], -8.207497747771947e-06, atol=1e-3, rtol=1e-3)
        assert_allclose(centroid[0][1], 1.989147771098717e-06, atol=1e-3, rtol=1e-3)

        assert_allclose(centroid[1][0], 3.069405792964239e-05, atol=1e-3, rtol=1e-3)
        assert_allclose(centroid[1][1], 12.421326489507168, atol=1e-3, rtol=1e-3)
        
        assert_allclose(centroid[2][0], 3.1631726815066986e-07, atol=1e-3, rtol=1e-3)
        assert_allclose(centroid[2][1], 18.13502264954927, atol=1e-3, rtol=1e-3)

    @patch("matplotlib.pyplot.show")
    def test_view_encircled_energy(self, mock_show, set_test_backend, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        encircled_energy.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_encircled_energy_larger_fig(self, mock_show, set_test_backend, cooke_triplet):
        encircled_energy = analysis.EncircledEnergy(cooke_triplet)
        encircled_energy.view(figsize=(20, 10))
        mock_show.assert_called_once()
        plt.close()


class TestCookeTripletRayFan:
    def test_ray_fan(self, set_test_backend, cooke_triplet):
        fan = analysis.RayFan(cooke_triplet)

        assert_allclose(fan.data["Px"][0], -1)
        assert_allclose(fan.data["Px"][-1], 1)

        assert_allclose(fan.data["Py"][0], -1)
        assert_allclose(fan.data["Py"][-1], 1)

        assert_allclose(fan.data["(0.0, 0.0)"]["0.48"]["x"][0], 0.00238814980958324, atol=1e-9, )

        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.48"]["x"][0],
            0.00238814980958324,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.48"]["x"][-1],
            -0.00238814980958324,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.48"]["y"][0],
            0.00238814980958324,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.48"]["y"][-1],
            -0.00238814980958324,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.55"]["x"][0],
            0.004195677081323623,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.55"]["x"][-1],
            -0.004195677081323623,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.55"]["y"][0],
            0.004195677081323623,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.55"]["y"][-1],
            -0.004195677081323623,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.65"]["x"][0],
            -8.284696919602652e-06,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.65"]["x"][-1],
            8.284696919602652e-06,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.65"]["y"][0],
            -8.284696919602652e-06,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.0)"]["0.65"]["y"][-1],
            8.284696919602652e-06,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.48"]["x"][0],
            0.01973142095198721,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.48"]["x"][-1],
            -0.01973142095198721,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.48"]["y"][0],
            -0.023207115035676296,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.48"]["y"][-1],
            0.03928464835618861,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.55"]["x"][0],
            0.021420191179537973,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.55"]["x"][-1],
            -0.021420191179537973,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.55"]["y"][0],
            -0.024812371459915994,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.55"]["y"][-1],
            0.04075295155640113,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.65"]["x"][0],
            0.017025487217305013,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.65"]["x"][-1],
            -0.017025487217305013,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.65"]["y"][0],
            -0.03229666187094615,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 0.7)"]["0.65"]["y"][-1],
            0.047721942006075935,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.48"]["x"][0],
            0.01563881685548374,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.48"]["x"][-1],
            -0.01563881685548374,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.48"]["y"][0],
            -0.0044989771745065354,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.48"]["y"][-1],
            0.013000385049824814,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.55"]["x"][0],
            0.016936433773790505,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.55"]["x"][-1],
            -0.016936433773790505,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.55"]["y"][0],
            -0.01705141007843025,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.55"]["y"][-1],
            0.022501847359645666,
            atol=1e-9,
        )

        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.65"]["x"][0],
            0.01214534602206907,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.65"]["x"][-1],
            -0.01214534602206907,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.65"]["y"][0],
            -0.033957537601747134,
            atol=1e-9,
        )
        assert_allclose(
            fan.data["(0.0, 1.0)"]["0.65"]["y"][-1],
            0.036545592330593735,
            atol=1e-9,
        )

    @patch("matplotlib.pyplot.show")
    def test_view_ray_fan(self, mock_show, set_test_backend, cooke_triplet):
        ray_fan = analysis.RayFan(cooke_triplet)
        ray_fan.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_ray_fan_larger_fig(self, mock_show, set_test_backend, cooke_triplet):
        ray_fan = analysis.RayFan(cooke_triplet)
        ray_fan.view(figsize=(20, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTelescopeTripletYYbar:
    @patch("matplotlib.pyplot.show")
    def test_view_yybar(self, mock_show, set_test_backend, telescope_objective):
        yybar = analysis.YYbar(telescope_objective)
        yybar.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_yybar_larger_fig(self, mock_show, set_test_backend, telescope_objective):
        yybar = analysis.YYbar(telescope_objective)
        yybar.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTelescopeTripletDistortion:
    def test_distortion_values(self, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective)

        assert_allclose(dist.data[0][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[0][-1], 0.005950509480884957, atol=1e-9)

        assert_allclose(dist.data[1][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[1][-1], 0.005786305783771451, atol=1e-9)

        assert_allclose(dist.data[0][0], 0.0, atol=1e-9)
        assert_allclose(dist.data[2][-1], 0.005720392850412076, atol=1e-9)

    def test_f_theta_distortion(self, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective, distortion_type="f-theta")

        assert_allclose(dist.data[0][0],0.0, atol=1e-9)
        assert_allclose(dist.data[0][-1],0.016106265133212852, atol=1e-9)
        
        assert_allclose(dist.data[1][0],0.0, atol=1e-9)
        assert_allclose(dist.data[1][-1],0.015942044760968603, atol=1e-9)
        
        assert_allclose(dist.data[2][0],0.0, atol=1e-9)
        assert_allclose(dist.data[2][-1],0.015876125134060767, atol=1e-9)

    def test_invalid_distortion_type(self, set_test_backend, telescope_objective):
        with pytest.raises(ValueError):
            analysis.Distortion(telescope_objective, distortion_type="invalid")

    @patch("matplotlib.pyplot.show")
    def test_view_distortion(self, mock_show, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        dist.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_distortion_larger_fig(self, mock_show, set_test_backend, telescope_objective):
        dist = analysis.Distortion(telescope_objective)
        dist.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTelescopeTripletGridDistortion:
    def test_grid_distortion_values(self, set_test_backend, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        assert_allclose(dist.data["max_distortion"], 0.005785718069180374, atol=1e-9)

        assert dist.data["xr"].shape == (10, 10)
        assert dist.data["yr"].shape == (10, 10)
        assert dist.data["xp"].shape == (10, 10)
        assert dist.data["yp"].shape == (10, 10)

        assert_allclose(dist.data["xr"][0, 0], -1.2342622299776145, atol=1e-9)
        assert_allclose(dist.data["xr"][4, 6], 0.41137984374933073, atol=1e-9)

        assert_allclose(dist.data["yr"][1, 0], -0.959951505834632, atol=1e-9)
        assert_allclose(dist.data["yr"][2, 6], -0.6856458243955965, atol=1e-9)

        assert_allclose(dist.data["xp"][0, 2], -0.6856375010477692, atol=1e-9)
        assert_allclose(dist.data["xp"][4, 4], -0.13712543741510327, atol=1e-9)

        assert_allclose(dist.data["yp"][-1, 0], 1.2341908231761498, atol=1e-9)
        assert_allclose(dist.data["yp"][1, 5], -0.9599069415493584, atol=1e-9)

    def test_f_theta_distortion(self, set_test_backend, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective, distortion_type="f-theta")

        assert_allclose(dist.data["max_distortion"], 0.010863278146924825, atol=1e-9)

        assert dist.data["xr"].shape == (10, 10)
        assert dist.data["yr"].shape == (10, 10)
        assert dist.data["xp"].shape == (10, 10)
        assert dist.data["yp"].shape == (10, 10)

        assert_allclose(dist.data["xr"][0, 0], -1.2342622299776145, atol=1e-9)
        assert_allclose(dist.data["xr"][4, 6], 0.41137984374933073, atol=1e-9)

        assert_allclose(dist.data["yr"][1, 0], -0.959951505834632, atol=1e-9)
        assert_allclose(dist.data["yr"][2, 6], -0.6856458243955965, atol=1e-9)

        assert_allclose(dist.data["xp"][0, 2], -0.6856267573347536, atol=1e-9)
        assert_allclose(dist.data["xp"][4, 4], -0.13712535146695065, atol=1e-9)

        assert_allclose(dist.data["yp"][-1, 0], 1.2341281632025562, atol=1e-9)
        assert_allclose(dist.data["yp"][1, 5], -0.9598774602686547, atol=1e-9)

    def test_invalid_distortion_type(self, set_test_backend, telescope_objective):
        with pytest.raises(ValueError):
            analysis.GridDistortion(telescope_objective, distortion_type="invalid")

    @patch("matplotlib.pyplot.show")
    def test_view_grid_distortion(self, mock_show, set_test_backend, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        dist.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_grid_distortion_larger_fig(self, mock_show, set_test_backend, telescope_objective):
        dist = analysis.GridDistortion(telescope_objective)
        dist.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestTelescopeTripletFieldCurvature:
    def test_field_curvature_init(self, set_test_backend, telescope_objective):
        field_curvature = analysis.FieldCurvature(telescope_objective)
        assert field_curvature.optic == telescope_objective
        assert (
            field_curvature.wavelengths
            == telescope_objective.wavelengths.get_wavelengths()
        )
        assert field_curvature.num_points == 128

    def test_field_curvature_init_with_wavelength(self, set_test_backend, telescope_objective):
        field_curvature = analysis.FieldCurvature(
            telescope_objective,
            wavelengths=[0.5, 0.6],
        )
        assert field_curvature.optic == telescope_objective
        assert field_curvature.wavelengths == [0.5, 0.6]
        assert field_curvature.num_points == 128

    def test_field_curvature_init_with_num_points(self, set_test_backend, telescope_objective):
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

    def test_field_curvature_init_with_all_parameters(self, set_test_backend, telescope_objective):
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
    def test_field_curvature_view(self, mock_show, set_test_backend, telescope_objective):
        field_curvature = analysis.FieldCurvature(telescope_objective)
        field_curvature.view()
        mock_show.assert_called_once()
        plt.close()

    def test_field_curvature_generate_data(self, set_test_backend, telescope_objective):
        f = analysis.FieldCurvature(telescope_objective)

        assert_allclose(f.data[0][0][89], -0.0013062169220806206, atol=1e-9)
        assert_allclose(f.data[0][1][40], 0.03435268469825703, atol=1e-9)
        assert_allclose(f.data[0][1][112], 0.012502083379998098, atol=1e-9)
        assert_allclose(f.data[0][0][81], 0.005363808856891348, atol=1e-9)
        assert_allclose(f.data[0][0][127], -0.041553105637156224, atol=1e-9)
        assert_allclose(f.data[0][0][40], 0.02969815644838593, atol=1e-9)
        assert_allclose(f.data[0][0][57], 0.021608994058848974, atol=1e-9)
        assert_allclose(f.data[0][1][45], 0.03350406866891282, atol=1e-9)
        assert_allclose(f.data[0][1][74], 0.026613511090172324, atol=1e-9)
        assert_allclose(f.data[0][1][94], 0.01990500178194723, atol=1e-9)

        assert_allclose(f.data[1][1][55], -0.004469963728211546, atol=1e-9)
        assert_allclose(f.data[1][1][19], 0.0008003571732224457, atol=1e-9)
        assert_allclose(f.data[1][1][93], -0.015595499139883678, atol=1e-9)
        assert_allclose(f.data[1][0][15], 0.0004226818372030349, atol=1e-9)
        assert_allclose(f.data[1][1][50], -0.0034313474749693047, atol=1e-9)
        assert_allclose(f.data[1][0][110], -0.05718858127937811, atol=1e-9)
        assert_allclose(f.data[1][0][89], -0.036917737894907106, atol=1e-9)
        assert_allclose(f.data[1][1][75], -0.00961346547634129, atol=1e-9)
        assert_allclose(f.data[1][0][69], -0.021587199726177217, atol=1e-9)

        assert_allclose(f.data[2][1][62], 0.059485399479466794, atol=1e-9)
        assert_allclose(f.data[2][0][103], 0.015768399161337723, atol=1e-9)
        assert_allclose(f.data[2][0][0], 0.06707048647659668, atol=1e-9)
        assert_allclose(f.data[2][1][68], 0.05794633552031286, atol=1e-9)
        assert_allclose(f.data[2][0][6], 0.06689636005684219, atol=1e-9)
        assert_allclose(f.data[2][1][40], 0.06391326892594748, atol=1e-9)
        assert_allclose(f.data[2][0][88], 0.029620344519446916, atol=1e-9)
        assert_allclose(f.data[2][0][5], 0.06694956529269887, atol=1e-9)
        assert_allclose(f.data[2][1][98], 0.048120430272662294, atol=1e-9)
        assert_allclose(f.data[2][0][5], 0.06694956529269887, atol=1e-9)


class TestSpotVsField:
    def test_rms_spot_size_vs_field_initialization(self, set_test_backend, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        assert spot_vs_field.num_fields == 64
        assert be.array_equal(spot_vs_field._field[:, 1], be.linspace(0, 1, 64))

    def test_rms_spot_radius(self, set_test_backend, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        spot_size = spot_vs_field._spot_size
        assert spot_size.shape == (
            64,
            len(telescope_objective.wavelengths.get_wavelengths()),
        )

    @patch("matplotlib.pyplot.show")
    def test_view_spot_vs_field(self, mock_show, set_test_backend, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        spot_vs_field.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_spot_vs_field_larger_fig(self, mock_show, set_test_backend, telescope_objective):
        spot_vs_field = analysis.RmsSpotSizeVsField(telescope_objective)
        spot_vs_field.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestWavefrontErrorVsField:
    def test_rms_wave_init(self, set_test_backend, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        assert wavefront_error_vs_field.num_fields == 32
        assert be.array_equal(
            wavefront_error_vs_field._field[:, 1],
            be.linspace(0, 1, 32),
        )

    def test_rms_wave(self, set_test_backend, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        wavefront_error = wavefront_error_vs_field._wavefront_error
        assert wavefront_error.shape == (
            32,
            len(telescope_objective.wavelengths.get_wavelengths()),
        )

    @patch("matplotlib.pyplot.show")
    def test_view_wave(self, mock_show, set_test_backend, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        wavefront_error_vs_field.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_wave_larger_fig(self, mock_show, set_test_backend, telescope_objective):
        wavefront_error_vs_field = analysis.RmsWavefrontErrorVsField(
            telescope_objective,
        )
        wavefront_error_vs_field.view(figsize=(12.4, 10))
        mock_show.assert_called_once()
        plt.close()


class TestPupilAberration:
    def test_initialization(self, set_test_backend, telescope_objective):
        pupil_ab = analysis.PupilAberration(telescope_objective)
        assert pupil_ab.optic == telescope_objective
        assert pupil_ab.fields == [(0.0, 0.0), (0.0, 0.7), (0.0, 1.0)]
        assert pupil_ab.wavelengths == [0.4861, 0.5876, 0.6563]
        assert pupil_ab.num_points == 257  # num_points is forced to be odd

    def test_generate_data(self, set_test_backend, telescope_objective):
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
    def test_view(self, mock_show, set_test_backend, telescope_objective):
        pupil_ab = analysis.PupilAberration(telescope_objective)
        pupil_ab.view()
        mock_show.assert_called_once()
        plt.close()

def test_spotdiagram_invalid_coordinates(cooke_triplet):
    with pytest.raises(ValueError) as excinfo:
        analysis.SpotDiagram(cooke_triplet, coordinates="invalid")
    assert str(excinfo.value) == "Coordinates must be 'global' or 'local'."

def test_generate_field_data_local(set_test_backend, cooke_triplet):

    spot = analysis.SpotDiagram(cooke_triplet, coordinates="local")

    # Pick the first field and wavelength
    field = spot.fields[0]
    wavelength = spot.wavelengths[0]

    data = spot._generate_field_data(
        field,
        wavelength,
        num_rays=10,
        distribution="hexapolar",
        coordinates="local",
    )

    plot_x, plot_y, _ = data
    global_x = spot.optic.surface_group.x[-1, :]
    global_y = spot.optic.surface_group.y[-1, :]
    assert_allclose(plot_x, global_x)
    assert_allclose(plot_y, global_y)
    
def test_generate_field_data_global(set_test_backend, cooke_triplet):

    spot = analysis.SpotDiagram(cooke_triplet, coordinates="global")

    # Pick the first field and wavelength
    field = spot.fields[0]
    wavelength = spot.wavelengths[0]

    data = spot._generate_field_data(
        field,
        wavelength,
        num_rays=10,
        distribution="hexapolar",
        coordinates="global",
    )

    plot_x, plot_y, _ = data
    global_x = spot.optic.surface_group.x[-1, :]
    global_y = spot.optic.surface_group.y[-1, :]
    assert_allclose(plot_x, global_x)
    assert_allclose(plot_y, global_y)
    
@pytest.fixture
def test_system_irradiance_v1():
    class TestSystemIrradianceV1(Optic):
        def __init__(self):
            super().__init__()
            self.add_surface(index=0, thickness=be.inf)
            self.add_surface(index=1, thickness=0, is_stop=True)
            self.add_surface(index=2, thickness=10)
            self.add_surface(index=3)  # image
            detector_size = RectangularAperture(x_max=2.5, x_min=-2.5, y_max=2.5, y_min=-2.5)
            self.surface_group.surfaces[-1].aperture = detector_size
            self.add_wavelength(0.55)
            self.set_field_type('angle')
            self.add_field(y=0)
            self.set_aperture('EPD', 5.0)
    return TestSystemIrradianceV1()

@pytest.fixture
def perfect_mirror_system():
    class PerfectMirror(Optic):
        def __init__(self):
            super().__init__()
            self.add_surface(index=0, thickness=be.inf)
            self.add_surface(index=1, thickness=50)
            self.add_surface(index=2, thickness=-25, radius=-50, conic=-1.0, material='mirror', is_stop=True)
            self.add_surface(index=3)  # image
            detector_size = RectangularAperture(x_max=2.5, x_min=-2.5, y_max=2.5, y_min=-2.5)
            self.surface_group.surfaces[-1].aperture = detector_size
            self.add_wavelength(0.55)
            self.set_field_type('angle')
            self.add_field(y=0)
            self.set_aperture('EPD', 5.0)
    return PerfectMirror()

def _create_square_grid_rays(num_rays_edge, min_coord, max_coord, wavelength_val=0.55):
    x_rays_np = be.linspace(min_coord, max_coord, num_rays_edge)
    x_np, y_np = be.meshgrid(x_rays_np, x_rays_np)
    x_be_flat = be.array(x_np.flatten())
    y_be_flat = be.array(y_np.flatten())
    num_rays = x_be_flat.shape[0]

    z_be = be.zeros((num_rays,)) # pass shape as tuple because of torch backend
    L_be = be.zeros((num_rays,)) 
    M_be = be.zeros((num_rays,)) 
    N_be = be.ones((num_rays,))  
    I_be = be.ones((num_rays,))  
    W_be = be.full((num_rays,), wavelength_val) 
    return RealRays(x_be_flat, y_be_flat, z_be, L_be, M_be, N_be, I_be, W_be)

def _apply_gaussian_apodization(x_coords_flat, y_coords_flat, sigma_x, sigma_y, peak_intensity=1.0):
    # x_coords_flat and y_coords_flat are expected to be 1D backend arrays
    x_np = be.to_numpy(x_coords_flat) 
    y_np = be.to_numpy(y_coords_flat)
    exponent = -(((x_np**2) / (2 * sigma_x**2)) + ((y_np**2) / (2 * sigma_y**2)))
    intensities_np = peak_intensity * be.exp(exponent)
    return be.array(intensities_np)


class TestIncoherentIrradiance:
    @patch("matplotlib.pyplot.show")
    def test_irradiance_v1_uniform_and_user_defined_rays(self, mock_show, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        res = (5, 5)

        # Test with default uniform rays
        irr_uniform = analysis.IncoherentIrradiance(optic_sys, num_rays=5, distribution='uniform', res=res) 
        irr_map_uniform, _, _ = irr_uniform.irr_data[0][0]
        
        # This is a basic check, not a precise value assertion
        assert be.sum(irr_map_uniform) > 0
        assert be.max(irr_map_uniform) > 0
        irr_uniform.view()
        plt.close()


        # Test with user-defined rays
        user_rays = _create_square_grid_rays(num_rays_edge=5, min_coord=-2.25, max_coord=1.75)
        irr_user = analysis.IncoherentIrradiance(optic_sys, res=res, user_initial_rays=user_rays)
        irr_map_user, _, _ = irr_user.irr_data[0][0]
    
        pixel_area_expected = ( (2.5 - (-2.5)) / res[0] ) * ( (2.5 - (-2.5)) / res[1] )
        expected_irr_value = 1.0 / pixel_area_expected
        assert_allclose(irr_map_user, be.full(res, expected_irr_value), atol=1e-5)
        irr_user.view()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_irradiance_v1_one_ray_per_other_pixel(self, mock_show, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        res_val = (10, 10)

        # Use be.linspace and be.meshgrid if available and compatible, otherwise numpy is fine for test setup
        x_centers_np = be.linspace(-2.25, 2.25, 10)
        selected_x_np = x_centers_np[::2]
        selected_y_np = x_centers_np[::2]
        x_np_mesh, y_np_mesh = be.meshgrid(selected_x_np, selected_y_np) # Use numpy for meshgrid setup
        
        x_be_flat = be.array(x_np_mesh.flatten())
        y_be_flat = be.array(y_np_mesh.flatten())
        num_rays_flat = x_be_flat.shape[0]

        user_rays = RealRays(
            x=x_be_flat,
            y=y_be_flat,
            z=be.zeros((num_rays_flat,)),
            L=be.zeros((num_rays_flat,)),
            M=be.zeros((num_rays_flat,)),
            N=be.ones((num_rays_flat,)),
            intensity=be.ones((num_rays_flat,)),      
            wavelength=be.full((num_rays_flat,), 0.55) 
        )

        irr_analysis = analysis.IncoherentIrradiance(optic_sys, res=res_val, user_initial_rays=user_rays)
        irr_map_be, _, _ = irr_analysis.irr_data[0][0]

        expected_map_np = np.zeros(res_val) # create the expected map with numpy 
        pixel_area_expected = ((2.5 - (-2.5)) / res_val[0]) * ((2.5 - (-2.5)) / res_val[1])
        irr_value_per_ray = 1.0 / pixel_area_expected

        for i in range(0, res_val[0], 2):
            for j in range(0, res_val[1], 2):
                expected_map_np[i, j] = irr_value_per_ray

        assert_allclose(irr_map_be, expected_map_np, atol=1e-5) # assert_allclose handles be_tensor vs np_array
        irr_analysis.view()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_irradiance_gaussian_apodization(self, mock_show, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        res_val = (50, 50)
        num_rays_edge = 100
        sigma_val = 0.5

        x_rays_np = be.linspace(-2.5, 2.5, num_rays_edge)
        x_np_mesh, y_np_mesh = be.meshgrid(x_rays_np, x_rays_np)
        x_be_flat = be.array(x_np_mesh.flatten())
        y_be_flat = be.array(y_np_mesh.flatten())
        num_rays_flat = x_be_flat.shape[0]

        gaussian_intensities = _apply_gaussian_apodization(x_be_flat, y_be_flat, sigma_x=sigma_val, sigma_y=sigma_val)

        user_rays_apodized = RealRays(
            x=x_be_flat, y=y_be_flat, z=be.zeros((num_rays_flat,)),
            L=be.zeros((num_rays_flat,)), M=be.zeros((num_rays_flat,)), N=be.ones((num_rays_flat,)),
            intensity=gaussian_intensities, wavelength=be.full((num_rays_flat,), 0.55)
        )

        irr_apodized = analysis.IncoherentIrradiance(optic_sys, res=res_val, user_initial_rays=user_rays_apodized)
        irr_map_apodized_be, x_edges, y_edges = irr_apodized.irr_data[0][0]

        center_idx_x = res_val[0] // 2
        center_idx_y = res_val[1] // 2
        
        val_center = irr_map_apodized_be[center_idx_x, center_idx_y]
        val_corner1 = irr_map_apodized_be[0, 0]
        val_corner2 = irr_map_apodized_be[res_val[0]-1, res_val[1]-1]
        max_val_be = be.max(irr_map_apodized_be)

        assert be.to_numpy(val_center) > be.to_numpy(val_corner1)
        assert be.to_numpy(val_center) > be.to_numpy(val_corner2)
        assert_allclose(val_center, max_val_be, rtol=1e-3) 

        irr_apodized.view()
        plt.close()


    @patch("matplotlib.pyplot.show")
    def test_irradiance_perfect_mirror_focus(self, mock_show, set_test_backend, perfect_mirror_system):
        optic_sys = perfect_mirror_system
        res_val = (21, 21)
        num_rays_epd = 51

        user_rays_grid = _create_square_grid_rays(num_rays_edge=num_rays_epd, min_coord=-2.5, max_coord=2.5)

        irr_perfect = analysis.IncoherentIrradiance(optic_sys, res=res_val, user_initial_rays=user_rays_grid)
        irr_map_perfect_be = irr_perfect.irr_data[0][0][0] 

        center_x_idx = res_val[0] // 2
        center_y_idx = res_val[1] // 2

        total_sum_be = be.sum(irr_map_perfect_be)
        center_pixel_value_be = irr_map_perfect_be[center_x_idx, center_y_idx]

        assert be.to_numpy(total_sum_be) > 1e-9 
        assert_allclose(center_pixel_value_be, total_sum_be, atol=1e-5)

        irr_map_perfect_np = be.to_numpy(irr_map_perfect_be) # convert for easy masking with numpy
        mask = np.ones(irr_map_perfect_np.shape, dtype=bool)
        mask[center_x_idx, center_y_idx] = False
        assert_allclose(np.sum(irr_map_perfect_np[mask]), 0.0, atol=1e-5)

        irr_perfect.view()
        plt.close()
        
    @patch("matplotlib.pyplot.show")
    def test_irradiance_plot_cross_section_gaussian(self, mock_show, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        res_val = (50, 50)
        num_rays_edge = 100
        sigma_val = 0.5

        x_rays_np = np.linspace(-2.5, 2.5, num_rays_edge)
        x_np_mesh, y_np_mesh = np.meshgrid(x_rays_np, x_rays_np)
        x_be_flat = be.array(x_np_mesh.flatten())
        y_be_flat = be.array(y_np_mesh.flatten())
        num_rays_flat = x_be_flat.shape[0]

        gaussian_intensities = _apply_gaussian_apodization(x_be_flat, y_be_flat, sigma_x=sigma_val, sigma_y=sigma_val)
        user_rays_apodized = RealRays(
            x=x_be_flat, y=y_be_flat, z=be.zeros((num_rays_flat,)),
            L=be.zeros((num_rays_flat,)), M=be.zeros((num_rays_flat,)), N=be.ones((num_rays_flat,)),
            intensity=gaussian_intensities, wavelength=be.full((num_rays_flat,), 0.55)
        )
        irr_apodized = analysis.IncoherentIrradiance(optic_sys, res=res_val, user_initial_rays=user_rays_apodized)

        # Test cross-X plot at center
        irr_apodized.view(cross_section=('cross-x', res_val[0] // 2))
        # Test cross-Y plot at default middle slice
        irr_apodized.view(cross_section=('cross-y', None))
        # Test cross-X plot with normalize=False
        irr_apodized.view(cross_section=('cross-x', res_val[0] // 2), normalize=False)

        assert mock_show.call_count >= 3
        plt.close('all') 

    @patch("matplotlib.pyplot.show")
    def test_irradiance_peak_irradiance(self, mock_show, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        irr = analysis.IncoherentIrradiance(optic_sys, num_rays=20, res=(10,10))
        peaks = irr.peak_irradiance() 
        assert len(peaks) == len(irr.fields)
        assert len(peaks[0]) == len(irr.wavelengths)
        assert be.to_numpy(peaks[0][0]) >= 0 

    def test_detector_surface_no_aperture(self, set_test_backend):
        class TestSystemNoAperture(Optic):
            def __init__(self):
                super().__init__()
                self.add_surface(index=0, thickness=be.inf)
                self.add_surface(index=1, thickness=0, is_stop=True)
                self.add_surface(index=2, thickness=10)
                self.add_surface(index=3) # Image plane, no aperture
                self.add_wavelength(0.55)
                self.set_field_type('angle')
                self.add_field(y=0)
                self.set_aperture('EPD', 5.0)
        optic_no_ap = TestSystemNoAperture()
        with pytest.raises(ValueError, match="Detector surface has no physical aperture"):
            analysis.IncoherentIrradiance(optic_no_ap, num_rays=5, res=(5,5))

    @patch("matplotlib.pyplot.show")
    @patch("builtins.print") 
    def test_px_size_overrides_res_warning(self, mock_print, mock_show, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        # Detector aperture is 5mm x 5mm. px_size of (1.0, 1.0) should result in 5x5 pixels.
        # res of (10,10) should be ignored and a warning printed.
        irr = analysis.IncoherentIrradiance(optic_sys, res=(10,10), px_size=(1.0, 1.0), num_rays=10)
        irr_map_be, x_edges, y_edges = irr.irr_data[0][0]

        # Check that the effective resolution is 5x5
        assert irr_map_be.shape == (5, 5)
        assert len(x_edges) == 5 + 1 # x_edges has N+1 elements for N pixels
        assert len(y_edges) == 5 + 1

        # Check that the warning was printed
        mock_print.assert_any_call("[IncoherentIrradiance] Warning: res parameter ignored - derived from px_size instead → (5,5) pixels")
        irr.view()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_irradiance_view_options(self, mock_show, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        irr = analysis.IncoherentIrradiance(optic_sys, num_rays=10, res=(10,10))

        # Test with different cmap and normalize=False
        irr.view(cmap="viridis", normalize=False)
        mock_show.assert_called_once()
        plt.close()
        mock_show.reset_mock() 

        # Test with cross-section, normalize=True, and different cmap
        irr.view(cross_section=('cross-x', 0), normalize=True, cmap="magma")
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    @patch("builtins.print")
    def test_irradiance_cross_section_invalid_slice(self, mock_print, mock_show, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        res_val = (5,5)
        irr = analysis.IncoherentIrradiance(optic_sys, num_rays=10, res=res_val)

        # Invalid slice index for cross-x
        irr.view(cross_section=('cross-x', res_val[0] + 5)) # Index out of bounds
        mock_print.assert_any_call(f"[IncoherentIrradiance] Warning: X-slice index {res_val[0]+5} is out of bounds for map shape {(res_val[0],res_val[1])}. Skipping plot.")
        
        # Invalid slice index for cross-y
        irr.view(cross_section=('cross-y', res_val[1] + 5)) # Index out of bounds
        mock_print.assert_any_call(f"[IncoherentIrradiance] Warning: Y-slice index {res_val[1]+5} is out of bounds for map shape {(res_val[0],res_val[1])}. Skipping plot.")

        # Invalid cross_section_info format (not tuple)
        irr.view(cross_section="invalid")
        mock_print.assert_any_call("[IncoherentIrradiance] Warning: Invalid cross_section_info type. Expected tuple. Defaulting to 2D plot.")
        # Invalid cross_section_info format (tuple wrong length)
        irr.view(cross_section=('cross-x',))
        mock_print.assert_any_call("[IncoherentIrradiance] Warning: Invalid cross_section_info type. Expected tuple. Defaulting to 2D plot.")
         # Invalid cross_section_info format (tuple wrong types)
        irr.view(cross_section=(123, 'cross-y'))
        mock_print.assert_any_call("[IncoherentIrradiance] Warning: Invalid cross_section_info format. Expected ('cross-x' or 'cross-y', int). Defaulting to 2D plot.")
        plt.close('all')


    @patch("matplotlib.pyplot.show")
    def test_irradiance_view_no_data(self, mock_show, capsys, set_test_backend, test_system_irradiance_v1):
        irr = analysis.IncoherentIrradiance(test_system_irradiance_v1, num_rays=1, res=(2,2)) # Minimal rays to get some data
        irr.irr_data = [] # Force no data
        irr.view()
        captured = capsys.readouterr()
        assert "No irradiance data to display." in captured.out
        mock_show.assert_not_called()
        plt.close()

        # Test with empty field block
        irr.irr_data = [[]]
        irr.view()
        captured = capsys.readouterr()
        assert "Warning: Field block 0 is empty." in captured.out or "No valid irradiance map data found to plot." in captured.out
        plt.close()

        # Test with None entry in field block
        irr.irr_data = [[None]]
        irr.view()
        captured = capsys.readouterr()
        assert "Warning: Entry 0 in field block 0 is None." in captured.out or "No valid irradiance map data found to plot." in captured.out
        plt.close()

        # Test with None irradiance map in entry
        dummy_edges = be.array([0.0,1.0]) 
        irr.irr_data = [[(None, dummy_edges, dummy_edges)]]
        irr.view()
        captured = capsys.readouterr()
        assert "Warning: Irradiance map in entry 0, field block 0 is None." in captured.out or "No valid irradiance map data found to plot." in captured.out
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_vmin_vmax_equal_case(self, mock_show, set_test_backend, test_system_irradiance_v1):
        optic_sys = test_system_irradiance_v1
        num_rays = 10
        user_rays = RealRays(
            x=be.full((num_rays,), 0.0), y=be.full((num_rays,), 0.0), z=be.zeros((num_rays,)),
            L=be.zeros((num_rays,)), M=be.zeros((num_rays,)), N=be.ones((num_rays,)),
            intensity=be.ones((num_rays,)), wavelength=be.full((num_rays,), 0.55)
        )
        res_tuple = (5,5)
        irr = analysis.IncoherentIrradiance(optic_sys, res=res_tuple, user_initial_rays=user_rays)
        dummy_edges = np.array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5]) # numpy array for dummy edges

        # Case 1: All pixels have same non-zero irradiance
        irr.irr_data = [[(be.full(res_tuple, 10.0), dummy_edges, dummy_edges)]]
        irr.view(normalize=False) # Test the vmin=vmax branch in plotting
        mock_show.assert_called_once()
        plt.close()
        mock_show.reset_mock()

        # Case 2: All pixels are zero
        irr.irr_data = [[(be.zeros(res_tuple), dummy_edges, dummy_edges)]]
        irr.view(normalize=False)
        mock_show.assert_called_once()
        plt.close()
        mock_show.reset_mock()

        # Case 3: Normalization active, vmin/vmax will be 0 and 1
        irr.irr_data = [[(be.full(res_tuple, 10.0), dummy_edges, dummy_edges)]]
        irr.view(normalize=True)
        mock_show.assert_called_once()
        plt.close()

    def test_peak_irradiance_empty_data(self, set_test_backend, test_system_irradiance_v1):
        irr = analysis.IncoherentIrradiance(test_system_irradiance_v1, num_rays=1, res=(2,2))
        irr.irr_data = []
        assert irr.peak_irradiance() == []

        irr.irr_data = [[]]
        assert irr.peak_irradiance() == [[]]

        dummy_edges = be.array([0., 1.]) # numpy array for dummy edges
        # Ensure float values for irradiance maps for consistent type with backend
        irr.irr_data = [[(be.array([[1.0,2.0],[3.0,4.0]]), dummy_edges, dummy_edges), 
                         (be.array([[5.0,6.0],[7.0,8.0]]), dummy_edges, dummy_edges)]]
        peaks = irr.peak_irradiance() # list of lists of backend scalars
        assert_allclose(peaks[0][0], be.array(4.0)) # Compare backend scalar with backend scalar
        assert_allclose(peaks[0][1], be.array(8.0))


def test_incoherent_irradiance_initialization(set_test_backend, test_system_irradiance_v1):
    optic = test_system_irradiance_v1
    irr = analysis.IncoherentIrradiance(optic, num_rays=10, res=(64, 64), px_size=(0.1, 0.1),
                               detector_surface=-1, fields="all", wavelengths="all",
                               distribution="random", user_initial_rays=None)
    assert irr.optic == optic
    assert irr.num_rays == 10
    # npix_x/y get overridden by px_size if px_size implies a different resolution
    # Detector is 5mm wide. 0.1mm pixels -> 50 pixels.
    assert irr.npix_x == 50 
    assert irr.npix_y == 50
    assert irr.px_size == (0.1, 0.1)
    assert irr.detector_surface == -1
    assert irr.fields == optic.fields.get_field_coords()
    assert irr.wavelengths == optic.wavelengths.get_wavelengths()
    assert irr.user_initial_rays is None
    assert len(irr.irr_data) == len(optic.fields.get_field_coords())
    assert len(irr.irr_data[0]) == len(optic.wavelengths.get_wavelengths())

@patch("matplotlib.pyplot.show")
def test_view_normalize_true_peak_zero(mock_show, set_test_backend, test_system_irradiance_v1):
    optic = test_system_irradiance_v1
    irr = analysis.IncoherentIrradiance(optic, num_rays=1, res=(5,5))
    dummy_edges = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) # numpy array for dummy
    irr.irr_data = [[(be.zeros((5,5)), dummy_edges, dummy_edges)]] # All zero irradiance map
    
    irr.view(normalize=True) # Should handle peak_val = 0
    mock_show.assert_called_once()
    plt.close()

@patch("matplotlib.pyplot.show")
@patch("builtins.print")
def test_cross_section_plot_helper_out_of_bounds(mock_print, mock_show, set_test_backend, test_system_irradiance_v1):
    optic = test_system_irradiance_v1
    irr = analysis.IncoherentIrradiance(optic, num_rays=5, res=(5,5))
    irr_map_be, x_edges, y_edges = irr.irr_data[0][0] # x_edges, y_edges are numpy arrays

    # Test cross-x out of bounds
    irr._plot_cross_section(irr_map_be, x_edges, y_edges, 'cross-x', 10, (6,5), "Test", True)
    mock_print.assert_any_call("[IncoherentIrradiance] Warning: X-slice index 10 is out of bounds for map shape (5, 5). Skipping plot.")
    
    # Test cross-y out of bounds
    irr._plot_cross_section(irr_map_be, x_edges, y_edges, 'cross-y', 10, (6,5), "Test", True)
    mock_print.assert_any_call("[IncoherentIrradiance] Warning: Y-slice index 10 is out of bounds for map shape (5, 5). Skipping plot.")

    # Test invalid axis type
    irr._plot_cross_section(irr_map_be, x_edges, y_edges, 'invalid-axis', 0, (6,5), "Test", True)
    