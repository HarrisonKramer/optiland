import pytest
import matplotlib

matplotlib.use("Agg")  # ensure non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
    def test_view_mtf_defaults(self, set_test_backend, optic):
        m = GeometricMTF(optic)
        fig, ax = m.view()  # default figsize, no reference overlay
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_mtf_custom_fig(self, set_test_backend, optic):
        m = GeometricMTF(optic)
        fig, ax = m.view(figsize=(20, 20), add_reference=True)
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_generate_data_scaled(self, set_test_backend, optic):
        m = GeometricMTF(optic, scale=True)
        m._generate_mtf_data()
        assert m.data is not None, "Scaled MTF data should be generated"

    def test_generate_data_unscaled(self, set_test_backend, optic):
        m = GeometricMTF(optic, scale=False)
        m._generate_mtf_data()
        assert m.data is not None, "Unscaled MTF data should be generated"

    def test_max_freq_specification(self, set_test_backend, optic):
        m1 = GeometricMTF(optic)

        wavelength = optic.primary_wavelength
        expected_cutoff = 1 / (wavelength * 1e-3 * optic.paraxial.FNO())
        assert be.to_numpy(m1.max_freq) == pytest.approx(be.to_numpy(expected_cutoff))

        custom_freq = 50.0
        m2 = GeometricMTF(optic, max_freq=custom_freq)
        assert be.to_numpy(m2.max_freq) == pytest.approx(custom_freq)


class TestFFTMTF:
    def test_view_mtf_defaults(self, set_test_backend, optic):
        m = FFTMTF(optic)
        fig, ax = m.view()
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_mtf_custom_fig(self, set_test_backend, optic):
        m = FFTMTF(optic)
        fig, ax = m.view(figsize=(20, 20), add_reference=True)
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

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

    @pytest.mark.parametrize(
        "num_rays,expected_pupil_sampling",
        [
            (32, 32),
            (64, 45),
            (128, 64),
            (256, 90),
            (1024, 181),
        ],
    )
    def test_num_rays_and_grid_size(
        self, set_test_backend, num_rays, expected_pupil_sampling, optic
    ):
        m = FFTMTF(optic, num_rays=num_rays, grid_size=None)

        assert m.num_rays == expected_pupil_sampling
        assert m.grid_size == 2 * num_rays
