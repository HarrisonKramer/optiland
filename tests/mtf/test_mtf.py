# tests/mtf/test_mtf.py
"""
Tests for the GeometricMTF and FFTMTF analysis tools.
"""
import pytest
import matplotlib

matplotlib.use("Agg")  # ensure non-interactive backend for testing
import matplotlib.pyplot as plt

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
    """Provides a fresh CookeTriplet instance for each test."""
    return CookeTriplet()


class TestGeometricMTF:
    """
    Tests the GeometricMTF class, which calculates the MTF based on a
    geometric spot diagram analysis.
    """

    def test_view_mtf_defaults(self, set_test_backend, optic):
        """
        Tests the view method with default parameters, ensuring it runs
        without error and returns a valid plot.
        """
        m = GeometricMTF(optic)
        fig, ax = m.view()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_view_mtf_custom_fig(self, set_test_backend, optic):
        """
        Tests the view method with custom parameters, such as figure size
        and adding a diffraction reference curve.
        """
        m = GeometricMTF(optic)
        fig, ax = m.view(figsize=(20, 20), add_reference=True)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_generate_data_scaled(self, set_test_backend, optic):
        """
        Tests that MTF data can be generated with scaling applied.
        """
        m = GeometricMTF(optic, scale=True)
        m._generate_mtf_data()
        assert m.data is not None

    def test_generate_data_unscaled(self, set_test_backend, optic):
        """
        Tests that MTF data can be generated without scaling.
        """
        m = GeometricMTF(optic, scale=False)
        m._generate_mtf_data()
        assert m.data is not None

    def test_max_freq_specification(self, set_test_backend, optic):
        """
        Tests that the maximum frequency is correctly calculated when set to
        'cutoff', and that a custom numeric value can also be used.
        """
        # Test default 'cutoff' calculation
        m1 = GeometricMTF(optic)
        wavelength = optic.primary_wavelength
        expected_cutoff = 1 / (wavelength * 1e-3 * optic.paraxial.FNO())
        assert be.to_numpy(m1.max_freq) == pytest.approx(be.to_numpy(expected_cutoff))

        # Test custom frequency
        custom_freq = 50.0
        m2 = GeometricMTF(optic, max_freq=custom_freq)
        assert be.to_numpy(m2.max_freq) == pytest.approx(custom_freq)


class TestFFTMTF:
    """
    Tests the FFTMTF class, which calculates the MTF from the Fourier
    transform of the Point Spread Function (PSF).
    """

    def test_view_mtf_defaults(self, set_test_backend, optic):
        """
        Tests the view method with default parameters.
        """
        m = FFTMTF(optic)
        fig, ax = m.view()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_view_mtf_custom_fig(self, set_test_backend, optic):
        """
        Tests the view method with custom parameters.
        """
        m = FFTMTF(optic)
        fig, ax = m.view(figsize=(20, 20), add_reference=True)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_generate_data_infinite_object(self, set_test_backend, optic):
        """
        Tests that MTF data is generated correctly for an object at infinity.
        """
        m = FFTMTF(optic)
        m._generate_mtf_data()
        assert hasattr(m, "mtf") and m.mtf is not None

    def test_generate_data_finite_object(self, set_test_backend, optic):
        """
        Tests that MTF data is generated correctly for a finite object distance.
        """
        optic.surface_group.surfaces[0].geometry.cs.z = be.array(1e6)
        m = FFTMTF(optic)
        m._generate_mtf_data()
        assert hasattr(m, "mtf") and m.mtf is not None

    @pytest.mark.parametrize(
        "num_rays,expected_pupil_sampling",
        [(32, 32), (64, 45), (128, 64), (256, 90), (1024, 181)]
    )
    def test_num_rays_and_grid_size(self, set_test_backend, num_rays, expected_pupil_sampling, optic):
        """
        Tests that the number of rays for pupil sampling and the FFT grid size
        are correctly determined based on the user's `num_rays` input.
        """
        m = FFTMTF(optic, num_rays=num_rays, grid_size=None)
        assert m.num_rays == expected_pupil_sampling
        assert m.grid_size == 2 * num_rays