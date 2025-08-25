from __future__ import annotations

from unittest.mock import patch, PropertyMock

import numpy as np
import pytest

import optiland.backend as be
from optiland.mtf import HuygensMTF
from optiland.samples.objectives import DoubleGauss


@pytest.fixture
def real_optic():
    """Use a real DoubleGauss optic with no mocking."""
    return DoubleGauss()


class TestHuygensMTF:
    def test_backend_not_numpy_raises(self, real_optic):
        with patch("optiland.backend.get_backend", return_value="torch"):  # noqa: SIM117
            with pytest.raises(ValueError, match="only supports the 'numpy' backend"):
                HuygensMTF(real_optic)

    def test_init_sets_attributes_cutoff(self, real_optic):
        with patch("optiland.backend.get_backend", return_value="numpy"):
            mtf = HuygensMTF(real_optic, max_freq="cutoff", image_size=16)
        assert mtf.num_rays == 128
        assert mtf.image_size == 16
        assert mtf.max_freq == pytest.approx(1 / (0.5876e-3 * mtf.FNO))
        assert isinstance(mtf.freq, be.ndarray)

    def test_init_non_primary_wavelength(self, real_optic):
        mtf = HuygensMTF(real_optic, max_freq="cutoff", image_size=16, wavelength=0.55)
        assert mtf.max_freq == pytest.approx(1 / (0.55e-3 * mtf.FNO))

    def test_init_with_numeric_max_freq(self, real_optic):
        mtf = HuygensMTF(real_optic, max_freq=200.0, image_size=8)
        assert mtf.max_freq == 200.0

    def test_get_fno(self, real_optic):
        mtf = HuygensMTF(real_optic, image_size=8)
        assert pytest.approx(4.992598838013766) == mtf.FNO

    def test_calculate_psf_stores_data(self, real_optic):
        mtf = HuygensMTF(real_optic, image_size=8)
        mtf.resolved_fields = [(0, 0)]
        mtf.resolved_wavelength = 0.55
        mtf._calculate_psf()
        assert len(mtf.psf_data) == 1
        assert mtf.psf_data[0].shape == (8, 8)
        assert len(mtf.psf_instances) == 1

    def test_generate_mtf_data_normalization(self, real_optic):
        mtf = HuygensMTF(real_optic, image_size=8)
        mtf.psf_data = [np.ones((8, 8))]
        results = mtf._generate_mtf_data()
        tangential, sagittal = results[0]
        assert np.all(tangential <= 1.0)
        assert np.all(sagittal <= 1.0)
        assert tangential[0] == 1.0
        assert sagittal[0] == 1.0

    def test_generate_mtf_data_divide_by_zero(self, real_optic):
        mtf = HuygensMTF(real_optic, image_size=8)
        mtf.psf_data = [np.zeros((8, 8))]
        results = mtf._generate_mtf_data()
        tangential, sagittal = results[0]
        assert np.allclose(tangential, 0)
        assert np.allclose(sagittal, 0)

    def test_plot_field_mtf_runs(self, real_optic):
        import matplotlib.pyplot as plt

        mtf = HuygensMTF(real_optic, image_size=8)
        mtf.resolved_fields = [(1.0, 2.0)]
        mtf.freq = np.arange(4)
        mtf_field_data = [np.linspace(0, 1, 4), np.linspace(1, 0, 4)]

        fig, ax = plt.subplots()
        mtf._plot_field_mtf(ax, 0, mtf_field_data, color="blue")

    def test_get_mtf_units_valid_and_invalid(self, real_optic):
        mtf = HuygensMTF(real_optic, image_size=8)
        mtf.resolved_fields = [(0, 0)]
        mtf.resolved_wavelength = 0.55
        mtf._calculate_psf()
        df = mtf._get_mtf_units()
        assert df > 0

        # Tamper with pixel pitch to trigger ValueError
        mtf.psf_instances[0].pixel_pitch = 0
        with pytest.raises(ValueError, match="Pixel pitch"):
            mtf._get_mtf_units()

    def test_integration_with_doublegauss(self, real_optic):
        """Full PSF + MTF calculation with a real optic (low resolution for speed)."""
        mtf = HuygensMTF(real_optic, fields=[(0, 0)], image_size=16)
        mtf._calculate_psf()
        mtf_data = mtf._generate_mtf_data()
        assert len(mtf_data) == 1
        tangential, sagittal = mtf_data[0]
        assert np.all((tangential >= 0) & (tangential <= 1))
        assert np.all((sagittal >= 0) & (sagittal <= 1))
