from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest

from optiland import mtf
from optiland.samples.objectives import CookeTriplet

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture(autouse=True)
def optic():
    return CookeTriplet()


class TestGeometricMTF:
    @patch("matplotlib.pyplot.show")
    def test_view_mtf(self, mock_show, optic):
        m = mtf.GeometricMTF(optic)
        m.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_mtf_larger_fig(self, mock_show, optic):
        m = mtf.GeometricMTF(optic)
        m.view(figsize=(20, 20), add_reference=True)
        mock_show.assert_called_once()
        plt.close()

    def test_generate_data(self, optic):
        m = mtf.GeometricMTF(optic)
        m._generate_mtf_data()
        assert m.data is not None

    def test_generate_data_no_scale(self, optic):
        m = mtf.GeometricMTF(optic, scale=False)
        m._generate_mtf_data()
        assert m.data is not None


class TestFFTMTF:
    @patch("matplotlib.pyplot.show")
    def test_view_mtf(self, mock_show, optic):
        m = mtf.FFTMTF(optic)
        m.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_mtf_larger_fig(self, mock_show, optic):
        m = mtf.FFTMTF(optic)
        m.view(figsize=(20, 20), add_reference=True)
        mock_show.assert_called_once()
        plt.close()

    def test_generate_data(self, optic):
        m = mtf.FFTMTF(optic)
        m._generate_mtf_data()
        assert m.mtf is not None

    def test_generate_data_finite_object(self, optic):
        optic.surface_group.surfaces[0].geometry.cs.z = 1e6
        m = mtf.FFTMTF(optic)
        m._generate_mtf_data()
        assert m.mtf is not None
