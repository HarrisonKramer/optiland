import pytest
import matplotlib

matplotlib.use("Agg")  # ensure non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch

import optiland.backend as be
from optiland.mtf import GeometricMTF, FFTMTF
from optiland.samples.objectives import CookeTriplet

# Parametrize every test over the two backends
pytestmark = pytest.mark.parametrize(
    "set_test_backend",
    ["numpy", "torch"],
    indirect=True,
    ids=["backend=numpy", "backend=torch"],
)


@pytest.fixture
def optic():
    """A fresh CookeTriplet for each test."""
    return CookeTriplet()


class TestGeometricMTF:
    @patch("matplotlib.pyplot.show")
    def test_view_mtf_defaults(self, mock_show, set_test_backend, optic):
        m = GeometricMTF(optic)
        m.view()  # default figsize, no reference overlay
        mock_show.assert_called_once()
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_view_mtf_custom_fig(self, mock_show, set_test_backend, optic):
        m = GeometricMTF(optic)
        m.view(figsize=(20, 20), add_reference=True)
        mock_show.assert_called_once()
        plt.close("all")

    def test_generate_data_scaled(self, set_test_backend, optic):
        m = GeometricMTF(optic, scale=True)
        m._generate_mtf_data()
        assert m.data is not None, "Scaled MTF data should be generated"

    def test_generate_data_unscaled(self, set_test_backend, optic):
        m = GeometricMTF(optic, scale=False)
        m._generate_mtf_data()
        assert m.data is not None, "Unscaled MTF data should be generated"


class TestFFTMTF:
    @patch("matplotlib.pyplot.show")
    def test_view_mtf_defaults(self, mock_show, set_test_backend, optic):
        m = FFTMTF(optic)
        m.view()
        mock_show.assert_called_once()
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_view_mtf_custom_fig(self, mock_show, set_test_backend, optic):
        m = FFTMTF(optic)
        m.view(figsize=(20, 20), add_reference=True)
        mock_show.assert_called_once()
        plt.close("all")

    def test_generate_data_infinite_object(self, set_test_backend, optic):
        """Default (infinite object distance) should produce an MTF array."""
        m = FFTMTF(optic)
        m._generate_mtf_data()
        assert hasattr(m, "mtf") and m.mtf is not None

    def test_generate_data_finite_object(self, set_test_backend, optic):
        """With a finite object distance, MTF still gets generated."""
        # Push the first surface very far away to mimic a finite-object scenario
        optic.surface_group.surfaces[0].geometry.cs.z = be.array(1e6)
        m = FFTMTF(optic)
        m._generate_mtf_data()
        assert hasattr(m, "mtf") and m.mtf is not None
