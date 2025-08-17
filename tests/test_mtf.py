import pytest
import matplotlib

matplotlib.use("Agg")  # ensure non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch

import warnings
import numpy as np

import optiland.backend as be
from optiland.optic import Optic
# ParaxialSurface class is not directly instantiated now, but good to know it exists
# from optiland.surfaces import ParaxialSurface
from optiland.geometries import Plane # Though not strictly needed if using surface_type="paraxial"
from optiland.materials import Ideal
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
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_view_mtf_custom_fig(self, set_test_backend, optic):
        m = GeometricMTF(optic)
        fig, ax = m.view(figsize=(20, 20), add_reference=True)
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
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
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_view_mtf_custom_fig(self, set_test_backend, optic):
        m = FFTMTF(optic)
        fig, ax = m.view(figsize=(20, 20), add_reference=True)
        assert fig is not None, "Figure should not be None"
        assert ax is not None, "Axes should not be None"
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
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


@pytest.fixture
def afocal_optic(set_test_backend): # Added set_test_backend to use be.inf correctly
    """Creates a simple afocal optical system."""
    optic = Optic(name="AfocalTestSystem")
    optic.set_field_type("angle")
    optic.add_field(y=0)
    optic.add_wavelength(value=0.55, is_primary=True)

    optic.add_surface(surface_type="object", material=Ideal())

    # Add ParaxialSurface with infinite focal length using surface_type
    optic.add_surface(
        surface_type="paraxial",
        focal_length=be.inf, # Use backend-aware infinity
        is_stop=True,
        material=Ideal() # Assumes material post this surface if not specified, or handled by factory
    )

    optic.add_surface(surface_type="image", material=Ideal())
    optic.set_aperture(aperture_type="EPD", value=10.0)

    optic.update()
    return optic


class TestAfocalSystemMTF:
    @patch("matplotlib.pyplot.show")
    def test_mtf_afocal_behavior(self, mock_show, set_test_backend, afocal_optic):
        """
        Tests MTF for an afocal system, checking warnings and behavior.
        """
        # --- Test GeometricMTF ---
        expected_warning_msg = "System FNO is inf.*MTF max_freq defaulted to 100.0 cycles/mm"
        with pytest.warns(UserWarning, match=expected_warning_msg) as record_geom:
            gmtf = GeometricMTF(afocal_optic, max_freq="cutoff")

        assert len(record_geom) == 1
        assert not any(w.category == RuntimeWarning for w in record_geom)

        assert gmtf.max_freq == pytest.approx(100.0)
        assert np.isinf(afocal_optic.paraxial.FNO())

        with warnings.catch_warnings(record=True) as record_gmtf_view:
            warnings.simplefilter("always")
            gmtf.view()
            assert len(record_gmtf_view) == 0, \
                f"gmtf.view() generated warnings: {[str(w.message) for w in record_gmtf_view]}"
        mock_show.assert_called_once()
        plt.close("all")
        mock_show.reset_mock()

        # --- Test FFTMTF ---
        # Expect two specific UserWarnings for FFTMTF
        with warnings.catch_warnings(record=True) as record_fft:
            warnings.simplefilter("always")
            fft_mtf = FFTMTF(afocal_optic, max_freq="cutoff", num_rays=32)

        raised_messages = [str(w.message) for w in record_fft]

        max_freq_warning_found = any(np.core.defchararray.match(expected_warning_msg, msg) for msg in raised_messages)
        mtf_unit_warning_found = any(np.core.defchararray.match("Calculated MTF frequency unit is infinite.*", msg) for msg in raised_messages)

        assert max_freq_warning_found, "FFTMTF max_freq default warning missing."
        assert mtf_unit_warning_found, "FFTMTF MTF unit warning missing."
        assert len(record_fft) == 2, f"Expected 2 warnings for FFTMTF, got {len(record_fft)}: {raised_messages}"

        assert not any(w.category == RuntimeWarning for w in record_fft)
        assert fft_mtf.max_freq == pytest.approx(100.0)

        with warnings.catch_warnings(record=True) as record_fft_view:
            warnings.simplefilter("always")
            fft_mtf.view()
            assert len(record_fft_view) == 0, \
                f"fft_mtf.view() generated warnings: {[str(w.message) for w in record_fft_view]}"

        mock_show.assert_called_once()
        plt.close("all")
