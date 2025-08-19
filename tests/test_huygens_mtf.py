from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import optiland.backend as be
from optiland.mtf.huygens import HuygensMTF
from optiland.samples.objectives import DoubleGauss


class DummyOptic:
    """Minimal dummy optic for testing _get_fno edge cases."""

    class Paraxial:
        def FNO(self):
            return 2.0

        def magnification(self):
            return 0.5

        def XPD(self):
            return 4.0

        def EPD(self):
            return 2.0

    class ObjectSurface:
        def __init__(self, infinite=True):
            self.is_infinite = infinite

    def __init__(self, infinite=True):
        self.paraxial = DummyOptic.Paraxial()
        self.object_surface = DummyOptic.ObjectSurface(infinite)
        self.primary_wavelength = 0.55  # um


@pytest.fixture
def dummy_optic():
    return DummyOptic()


class TestHuygensMTF:
    def test_backend_not_numpy_raises(self, dummy_optic):
        with patch("optiland.backend.get_backend", return_value="torch"):  # noqa: SIM117
            with pytest.raises(ValueError, match="only supports the 'numpy' backend"):
                HuygensMTF(dummy_optic)

    def test_init_sets_attributes_cutoff(self, dummy_optic):
        with patch("optiland.backend.get_backend", return_value="numpy"):
            mtf = HuygensMTF(dummy_optic, max_freq="cutoff")
        assert mtf.num_rays == 128
        assert mtf.image_size == 128
        assert mtf.max_freq == pytest.approx(1 / (0.55e-3 * mtf.FNO))
        assert isinstance(mtf.freq, be.ndarray)

    def test_init_with_numeric_max_freq(self, dummy_optic):
        mtf = HuygensMTF(dummy_optic, max_freq=200.0)
        assert mtf.max_freq == 200.0

    @pytest.mark.parametrize("infinite", [True, False])
    def test_get_fno_infinite_vs_finite(self, infinite):
        optic = DummyOptic(infinite=infinite)
        mtf = HuygensMTF(optic)
        if infinite:
            assert optic.paraxial.FNO() == mtf.FNO
        else:
            assert optic.paraxial.FNO() != mtf.FNO

    def test_get_fno_zero_epd(self, dummy_optic):
        dummy_optic.paraxial.EPD = lambda: 0
        mtf = HuygensMTF(dummy_optic)
        assert dummy_optic.paraxial.FNO() == mtf.FNO

    def test_calculate_psf_stores_data(self, dummy_optic):
        fake_psf = np.ones((8, 8))
        with patch("optiland.psf.huygens_fresnel.HuygensPSF") as MockPSF:
            instance = MagicMock()
            instance.psf = fake_psf
            instance.pixel_pitch = 0.01
            MockPSF.return_value = instance

            mtf = HuygensMTF(dummy_optic, image_size=8)
            mtf.resolved_fields = [(0, 0)]
            mtf.resolved_wavelength = 0.55
            mtf._calculate_psf()

            assert len(mtf.psf_data) == 1
            assert np.allclose(mtf.psf_data[0], fake_psf)
            assert len(mtf.psf_instances) == 1

    def test_generate_mtf_data_normalization(self, dummy_optic):
        mtf = HuygensMTF(dummy_optic, image_size=8)
        mtf.psf_data = [np.ones((8, 8))]
        results = mtf._generate_mtf_data()
        tangential, sagittal = results[0]
        assert np.all(tangential <= 1.0)
        assert np.all(sagittal <= 1.0)
        assert tangential[0] == 1.0
        assert sagittal[0] == 1.0

    def test_generate_mtf_data_divide_by_zero(self, dummy_optic):
        mtf = HuygensMTF(dummy_optic, image_size=8)
        psf = np.zeros((8, 8))
        mtf.psf_data = [psf]
        results = mtf._generate_mtf_data()
        tangential, sagittal = results[0]
        assert np.allclose(tangential, 0)
        assert np.allclose(sagittal, 0)

    def test_plot_field_mtf_runs(self, dummy_optic):
        import matplotlib.pyplot as plt

        mtf = HuygensMTF(dummy_optic, image_size=8)
        mtf.resolved_fields = [(1.0, 2.0)]
        mtf.freq = np.arange(4)
        mtf_field_data = [np.linspace(0, 1, 4), np.linspace(1, 0, 4)]

        fig, ax = plt.subplots()
        mtf._plot_field_mtf(ax, 0, mtf_field_data, color="blue")

    def test_get_mtf_units_valid_and_invalid(self, dummy_optic):
        fake_psf = np.ones((8, 8))
        with patch("optiland.psf.huygens_fresnel.HuygensPSF") as MockPSF:
            instance = MagicMock()
            instance.psf = fake_psf
            instance.pixel_pitch = 0.01
            MockPSF.return_value = instance

            mtf = HuygensMTF(dummy_optic, image_size=8)
            mtf.resolved_fields = [(0, 0)]
            mtf.resolved_wavelength = 0.55
            mtf._calculate_psf()
            df = mtf._get_mtf_units()
            assert df == pytest.approx(1 / (8 * 0.01))

            # Test invalid pixel_pitch
            mtf.psf_instances[0].pixel_pitch = 0
            with pytest.raises(ValueError, match="Pixel pitch"):
                mtf._get_mtf_units()

    def test_integration_with_doublegauss(self):
        """Runs a full PSF + MTF calculation with a real optic (slow)."""
        optic = DoubleGauss()
        mtf = HuygensMTF(optic, fields=[(0, 0)], image_size=64)
        mtf._calculate_psf()
        mtf_data = mtf._generate_mtf_data()
        assert len(mtf_data) == 1
        tangential, sagittal = mtf_data[0]
        assert np.all((tangential >= 0) & (tangential <= 1))
        assert np.all((sagittal >= 0) & (sagittal <= 1))
