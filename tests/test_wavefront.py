from unittest.mock import patch
import pytest
import numpy as np
from optiland import wavefront, distribution
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland.samples.eyepieces import EyepieceErfle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # use non-interactive backend for testing


class TestWavefront:
    @pytest.mark.parametrize('optic', [CookeTriplet(), DoubleGauss(),
                                       EyepieceErfle()])
    def test_wavefront_initialization(self, optic):
        w = wavefront.Wavefront(optic)
        assert w.num_rays == 12
        assert w.fields == optic.fields.get_field_coords()
        assert w.wavelengths == optic.wavelengths.get_wavelengths()
        assert isinstance(w.distribution, distribution.HexagonalDistribution)

    def test_wavefront_init_custom(self):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic, num_rays=100,
                                distribution='random', wavelengths='primary')
        assert w.num_rays == 100
        assert isinstance(w.distribution, distribution.RandomDistribution)
        assert w.wavelengths == [optic.primary_wavelength]

    def test_generate_data(self):
        optic = EyepieceErfle()
        w = wavefront.Wavefront(optic)
        data = w._generate_data(w.fields, w.wavelengths)
        assert len(data) == 3
        assert isinstance(data, list)
        assert isinstance(data[0], list)
        assert isinstance(data[0][0], tuple)
        assert isinstance(data[0][0][0], np.ndarray)
        assert w.data[0][0][0].size == 469  # num points in the pupil

    def test_trace_chief_ray(self):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        w._trace_chief_ray((0, 0), 0.55)
        assert np.all(optic.surface_group.y == 0)

    def test_get_reference_sphere(self):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        w._trace_chief_ray((0, 0), 0.55)
        xc, yc, zc, R = w._get_reference_sphere(pupil_z=100)
        assert np.allclose(xc, np.array([0.0]))
        assert np.allclose(yc, np.array([0.0]))
        assert np.allclose(zc, np.array([139.454938]))
        assert np.allclose(R, np.array([39.454938]))

    def test_ger_reference_sphere_error(self):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        optic.trace(Hx=0, Hy=0, wavelength=0.55)
        # fails when >1 rays traced in the pupil
        with pytest.raises(ValueError):
            w._get_reference_sphere(pupil_z=100)

    def test_get_path_length(self):
        optic = CookeTriplet()
        w = wavefront.Wavefront(optic)
        w._trace_chief_ray((0, 0), 0.55)
        xc, yc, zc, R = w._get_reference_sphere(pupil_z=100)
        path_length = w._get_path_length(xc, yc, zc, R)
        assert np.allclose(path_length, np.array([27.41117881]))

    def test_correct_tilt(self):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        opd = np.linspace(5, 100, w.distribution.x.size)
        corrected_opd = w._correct_tilt((0, 1), opd, x=None, y=None)
        assert np.isclose(corrected_opd[0], 2.5806903748015824)
        assert np.isclose(corrected_opd[10], 5.013823175582515)
        assert np.isclose(corrected_opd[100], 24.08949048654609)
        assert np.isclose(corrected_opd[111], 24.699015344473096)
        assert np.isclose(corrected_opd[242], 52.123070395591235)

    def test_opd_image_to_xp(self):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        w._trace_chief_ray((0, 0), 0.55)
        xc, yc, zc, R = w._get_reference_sphere(pupil_z=100)
        t = w._opd_image_to_xp(xc, yc, zc, R)
        assert np.allclose(t, np.array([39.454938]))


class TestOPDFan:
    def test_opd_fan_initialization(self):
        optic = DoubleGauss()
        opd_fan = wavefront.OPDFan(optic)
        assert opd_fan.num_rays == 100
        assert opd_fan.fields == optic.fields.get_field_coords()
        assert opd_fan.wavelengths == optic.wavelengths.get_wavelengths()
        assert isinstance(opd_fan.distribution,
                          distribution.CrossDistribution)
        arr = np.linspace(-1, 1, opd_fan.num_rays)
        assert np.all(opd_fan.pupil_coord == arr)

    @patch('matplotlib.pyplot.show')
    def test_opd_fan_view(self, moch_show):
        optic = DoubleGauss()
        opd_fan = wavefront.OPDFan(optic)
        opd_fan.view()
        moch_show.assert_called_once()
        plt.close()

    @patch('matplotlib.pyplot.show')
    def test_opd_fan_view_large(self, moch_show):
        optic = DoubleGauss()
        opd_fan = wavefront.OPDFan(optic)
        opd_fan.view(figsize=(20, 20))
        moch_show.assert_called_once()
        plt.close()
