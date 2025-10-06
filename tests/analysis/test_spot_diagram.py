# tests/analysis/test_spot_diagram.py
"""
Tests for the SpotDiagram and ThroughFocusSpotDiagram analysis tools.
"""
import matplotlib
import matplotlib.pyplot as plt
import pytest

import optiland.backend as be
from optiland import analysis
from optiland.optic import Optic
from optiland.samples.objectives import CookeTriplet
from ..utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def cooke_triplet():
    """Provides a CookeTriplet instance for testing."""
    return CookeTriplet()


@pytest.fixture
def triplet_four_fields():
    """Provides a Triplet lens with four field points for testing."""
    lens = Optic()
    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(index=1, radius=22.01359, thickness=3.25896, material="SK16")
    lens.add_surface(index=2, radius=-435.76044, thickness=6.00755)
    lens.add_surface(index=3, radius=-22.21328, thickness=0.99997, material=("F2", "schott"))
    lens.add_surface(index=4, radius=20.29192, thickness=4.75041, is_stop=True)
    lens.add_surface(index=5, radius=79.68360, thickness=2.95208, material="SK16")
    lens.add_surface(index=6, radius=-18.39533, thickness=42.20778)
    lens.add_surface(index=7)
    lens.set_aperture(aperture_type="EPD", value=10)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0); lens.add_field(y=10); lens.add_field(y=15); lens.add_field(y=20)
    lens.add_wavelength(value=0.48); lens.add_wavelength(value=0.55, is_primary=True); lens.add_wavelength(value=0.65)
    lens.update_paraxial()
    return lens


class TestCookeTripetSpotDiagram:
    """Tests the SpotDiagram analysis for the Cooke Triplet lens."""

    def test_spot_geometric_radius(self, set_test_backend, cooke_triplet):
        """Tests the calculation of the geometric spot radius."""
        spot = analysis.SpotDiagram(cooke_triplet)
        geo_radius = spot.geometric_spot_radius()
        assert_allclose(geo_radius[0][0], 0.005972, atol=1e-5)
        assert_allclose(geo_radius[1][1], 0.038646, atol=1e-5)
        assert_allclose(geo_radius[2][2], 0.037470, atol=1e-5)

    def test_spot_rms_radius(self, set_test_backend, cooke_triplet):
        """Tests the calculation of the RMS spot radius."""
        spot = analysis.SpotDiagram(cooke_triplet)
        rms_radius = spot.rms_spot_radius()
        assert_allclose(rms_radius[0][0], 0.003791, atol=1e-5)
        assert_allclose(rms_radius[1][1], 0.016786, atol=1e-5)
        assert_allclose(rms_radius[2][2], 0.013596, atol=1e-5)

    def test_airy_disc(self, set_test_backend, cooke_triplet):
        """Tests the calculation of the Airy disc radius."""
        spot = analysis.SpotDiagram(cooke_triplet)
        airy_radius_x, airy_radius_y = spot.airy_disc_x_y(wavelength=cooke_triplet.primary_wavelength)
        assert_allclose(airy_radius_x[0], 0.003340, atol=1e-5)
        assert_allclose(airy_radius_y[2], 0.003545, atol=1e-5)

    def test_view_spot_diagram(self, set_test_backend, cooke_triplet):
        """Tests the view method for generating a spot diagram plot."""
        spot = analysis.SpotDiagram(cooke_triplet)
        fig, axes = spot.view(add_airy_disk=True)
        assert fig is not None
        assert len(axes) > 0
        plt.close(fig)


class TestTripletSpotDiagram:
    """Tests the SpotDiagram analysis for a triplet with four fields."""

    def test_view_spot_diagram(self, set_test_backend, triplet_four_fields):
        """Tests the view method for a more complex field setup."""
        spot = analysis.SpotDiagram(triplet_four_fields)
        fig, axes = spot.view(figsize=(20, 10))
        assert fig is not None
        assert len(axes) > 0
        plt.close(fig)


class TestThroughFocusSpotDiagram:
    """Tests the ThroughFocusSpotDiagram analysis."""

    @pytest.fixture
    def tf_spot(self, set_test_backend):
        """Provides a ThroughFocusSpotDiagram instance for testing."""
        return analysis.ThroughFocusSpotDiagram(CookeTriplet(), delta_focus=0.05, num_steps=3, fields="all", wavelengths="all")

    def test_init_valid(self, set_test_backend):
        """Tests the initialization of the analysis tool."""
        tf = analysis.ThroughFocusSpotDiagram(CookeTriplet())
        assert tf.coordinates == "local"
        assert tf.distribution == "hexapolar"

    def test_init_invalid_coordinates(self, set_test_backend):
        """Tests that an invalid coordinate system raises a ValueError."""
        with pytest.raises(ValueError, match="Coordinates must be 'global' or 'local'"):
            analysis.ThroughFocusSpotDiagram(CookeTriplet(), coordinates="invalid")

    def test_analysis_results_content(self, set_test_backend, cooke_triplet):
        """
        Verifies the RMS spot size results at different focus positions against
        a direct calculation at nominal focus.
        """
        tf_spot = analysis.ThroughFocusSpotDiagram(cooke_triplet, delta_focus=0.05, num_steps=1)
        nominal_results_dict = next(res for res in tf_spot.results if be.isclose(list(res.keys())[0], 0.0))
        rms_values_at_nominal = list(nominal_results_dict.values())[0]

        spot_direct = analysis.SpotDiagram(cooke_triplet)
        rms_direct_all_wl = spot_direct.rms_spot_radius()
        primary_wl_idx = cooke_triplet.wavelengths.primary_index
        expected_rms_at_nominal = [rms_direct_all_wl[i][primary_wl_idx] for i in range(len(tf_spot.fields))]

        assert_allclose(be.to_numpy(rms_values_at_nominal), be.to_numpy(expected_rms_at_nominal))

    def test_view_full(self, tf_spot):
        """Tests the main view method to ensure it runs without error."""
        fig, axes = tf_spot.view()
        assert fig is not None
        assert axes is not None
        plt.close(fig)