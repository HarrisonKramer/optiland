# tests/mtf/test_huygens_mtf.py
"""
Tests for the HuygensMTF analysis tool in optiland.mtf.
"""
from __future__ import annotations
from unittest.mock import patch
import numpy as np
import pytest
import matplotlib

import optiland.backend as be
from optiland.mtf import HuygensMTF
from optiland.samples.objectives import DoubleGauss

matplotlib.use("Agg") # Use non-interactive backend for testing


@pytest.fixture
def real_optic():
    """Provides a real DoubleGauss optic instance for testing."""
    return DoubleGauss()


class TestHuygensMTF:
    """
    Tests the HuygensMTF class, which calculates the Modulation Transfer
    Function (MTF) from a Point Spread Function (PSF) generated via
    Huygens' principle.
    """

    def test_backend_not_numpy_raises(self, real_optic):
        """
        Ensures that initializing HuygensMTF raises a ValueError if the
        backend is not set to 'numpy'.
        """
        with patch("optiland.backend.get_backend", return_value="torch"):
            with pytest.raises(ValueError, match="only supports the 'numpy' backend"):
                HuygensMTF(real_optic)

    def test_init_sets_attributes_cutoff(self, real_optic):
        """
        Tests that the class initializes correctly and calculates the cutoff
        frequency when `max_freq` is set to 'cutoff'.
        """
        with patch("optiland.backend.get_backend", return_value="numpy"):
            mtf = HuygensMTF(real_optic, max_freq="cutoff", image_size=16)
        assert mtf.num_rays == 128
        assert mtf.image_size == 16
        assert mtf.max_freq == pytest.approx(1 / (0.5876e-3 * mtf.FNO))

    def test_init_with_numeric_max_freq(self, real_optic):
        """
        Tests that a numeric `max_freq` value is correctly assigned during
        initialization.
        """
        mtf = HuygensMTF(real_optic, max_freq=200.0, image_size=8)
        assert mtf.max_freq == 200.0

    def test_get_fno(self, real_optic):
        """Tests the calculation of the working F-number (FNO)."""
        mtf = HuygensMTF(real_optic, image_size=8)
        assert mtf.FNO == pytest.approx(4.9925988)

    def test_calculate_psf_stores_data(self, real_optic):
        """
        Tests that the internal `_calculate_psf` method correctly generates
        and stores PSF data.
        """
        mtf = HuygensMTF(real_optic, image_size=8)
        mtf.resolved_fields = [(0, 0)]
        mtf.resolved_wavelength = 0.55
        mtf._calculate_psf()
        assert len(mtf.psf_data) == 1
        assert mtf.psf_data[0].shape == (8, 8)

    def test_generate_mtf_data_normalization(self, real_optic):
        """
        Tests that the generated MTF data is correctly normalized, with a
        value of 1.0 at zero frequency.
        """
        mtf = HuygensMTF(real_optic, image_size=8)
        mtf.psf_data = [np.ones((8, 8))]  # Use a dummy PSF
        results = mtf._generate_mtf_data()
        tangential, sagittal = results[0]
        assert np.all(tangential <= 1.0)
        assert tangential[0] == 1.0

    def test_generate_mtf_data_divide_by_zero(self, real_optic):
        """
        Tests that if the PSF data is all zeros (e.g., no light), the MTF
        is correctly calculated as all zeros, avoiding division-by-zero errors.
        """
        mtf = HuygensMTF(real_optic, image_size=8)
        mtf.psf_data = [np.zeros((8, 8))]
        results = mtf._generate_mtf_data()
        tangential, sagittal = results[0]
        assert np.allclose(tangential, 0)

    def test_plot_field_mtf_runs(self, real_optic):
        """
        Smoke test for the internal `_plot_field_mtf` method to ensure it
        runs without error.
        """
        import matplotlib.pyplot as plt
        mtf = HuygensMTF(real_optic, image_size=8)
        mtf.resolved_fields = [(1.0, 2.0)]
        mtf.freq = np.arange(4)
        mtf_field_data = [np.linspace(0, 1, 4), np.linspace(1, 0, 4)]
        fig, ax = plt.subplots()
        mtf._plot_field_mtf(ax, 0, mtf_field_data, color="blue")
        plt.close(fig)

    def test_integration_with_doublegauss(self, real_optic):
        """
        Performs a full integration test by calculating the PSF and MTF for a
        real optical system.
        """
        mtf = HuygensMTF(real_optic, fields=[(0, 0)], image_size=16)
        mtf._calculate_psf()
        mtf_data = mtf._generate_mtf_data()
        assert len(mtf_data) == 1
        tangential, sagittal = mtf_data[0]
        assert np.all((tangential >= 0) & (tangential <= 1))
        assert np.all((sagittal >= 0) & (sagittal <= 1))