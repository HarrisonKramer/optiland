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


from unittest.mock import MagicMock
from optiland.mtf import HuygensMTF # Import the new class
from optiland.optic import Optic # For creating a simple optic for finite object test
from optiland.surfaces import StandardSurface # For finite object test
from optiland.materials import Material # For finite object test


class TestHuygensMTF:
    @pytest.fixture(autouse=True)
    def ensure_numpy_backend(self, current_backend):
        """Fixture to ensure numpy backend is used for these tests."""
        if be.get_backend() != "numpy":
            be.set_backend("numpy")
        # No need to yield or teardown if current_backend fixture handles restoration

    @pytest.fixture
    def optic_fixture(self):
        """Provides a fresh CookeTriplet for each test method in this class."""
        return CookeTriplet()

    @patch("matplotlib.pyplot.show")
    def test_view_mtf_defaults(self, mock_show, optic_fixture, ensure_numpy_backend):
        """Test HuygensMTF.view() with default arguments."""
        # be.set_backend("numpy") # Ensure numpy backend
        m = HuygensMTF(optic_fixture, num_rays=32, image_size=32) # Use low res for speed
        m.view()
        mock_show.assert_called_once()
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_view_mtf_custom_fig(self, mock_show, optic_fixture, ensure_numpy_backend):
        """Test HuygensMTF.view() with custom figure size and reference."""
        # be.set_backend("numpy")
        m = HuygensMTF(optic_fixture, num_rays=32, image_size=32)
        m.view(figsize=(10, 5), add_reference=True) # Adjusted figsize for typical MTF
        mock_show.assert_called_once()
        plt.close("all")

    def test_generate_data_infinite_object(self, optic_fixture, ensure_numpy_backend):
        """Test MTF data generation for infinite object distance (default)."""
        # be.set_backend("numpy")
        m = HuygensMTF(optic_fixture, num_rays=32, image_size=32)
        # _generate_mtf_data is called during __init__ by BaseMTF
        assert hasattr(m, "mtf"), "MTF data should be generated during init."
        assert m.mtf is not None, "MTF data should not be None."
        assert len(m.mtf) == len(m.resolved_fields), "MTF data list length should match resolved fields."
        if m.mtf: # If there are fields and MTF data
            assert len(m.mtf[0]) == 2, "Each field's MTF data should have tangential and sagittal components."
            assert m.mtf[0][0] is not None, "Tangential MTF data should not be None."
            assert m.mtf[0][1] is not None, "Sagittal MTF data should not be None."
            # Check that MTF values are within [0, 1]
            assert be.all(m.mtf[0][0] >= 0) and be.all(m.mtf[0][0] <= 1.001) # Allow small tolerance
            assert be.all(m.mtf[0][1] >= 0) and be.all(m.mtf[0][1] <= 1.001)


    def test_generate_data_finite_object(self, ensure_numpy_backend):
        """Test MTF data generation for a finite object distance."""
        # be.set_backend("numpy")
        # Create a simple optic setup for finite object distance
        finite_optic = Optic()
        finite_optic.object_surface.is_infinite = False
        finite_optic.object_surface.geometry.cs.z = be.array(200.0) # Finite object distance
        # Add a lens (minimal setup for paraxial calculations to work)
        finite_optic.add_surface(StandardSurface(radius=50, thickness=5, material=Material("N-BK7")))
        finite_optic.add_surface(StandardSurface(radius=-50, thickness=100)) # Image plane
        finite_optic.update()

        m = HuygensMTF(finite_optic, fields=[(0,0)], wavelength=0.55, num_rays=32, image_size=32)
        assert hasattr(m, "mtf") and m.mtf is not None
        assert len(m.mtf) == 1
        assert len(m.mtf[0]) == 2
        assert be.all(m.mtf[0][0] >= 0) and be.all(m.mtf[0][0] <= 1.001)
        assert be.all(m.mtf[0][1] >= 0) and be.all(m.mtf[0][1] <= 1.001)


    def test_backend_check_raises_error(self, optic_fixture, current_backend):
        """Test HuygensMTF raises ValueError if backend is not numpy."""
        original_backend = be.get_backend()
        if original_backend == "torch": # If already torch, just try to init
            with pytest.raises(ValueError, match="HuygensMTF only supports the 'numpy' backend"):
                HuygensMTF(optic_fixture, num_rays=32, image_size=32)
        else: # If numpy or other, switch to torch, test, then switch back
            try:
                be.set_backend("torch")
                # Mock HuygensPSF to prevent its own backend check if it runs before HuygensMTF's
                with patch("optiland.psf.huygens_fresnel.HuygensPSF") as mock_psf:
                    # Ensure the mock_psf doesn't try to do things that require torch setup
                    mock_psf_instance = MagicMock()
                    mock_psf_instance.psf = be.zeros((32,32)) # dummy psf data
                    mock_psf_instance.pixel_pitch = 0.01
                    mock_psf.return_value = mock_psf_instance

                    with pytest.raises(ValueError, match="HuygensMTF only supports the 'numpy' backend"):
                        HuygensMTF(optic_fixture, num_rays=32, image_size=32)
            finally:
                be.set_backend(original_backend) # Restore original backend

    def test_max_freq_cutoff(self, optic_fixture, ensure_numpy_backend):
        """Test if max_freq is calculated correctly when set to 'cutoff'."""
        # be.set_backend("numpy")
        m = HuygensMTF(optic_fixture, num_rays=32, image_size=32, max_freq="cutoff")

        # Expected max_freq = 1 / (wavelength_um * 1e-3 * FNO)
        # FNO calculation can be complex, so we check it's positive and finite.
        # optic_fixture.primary_wavelength is in um
        # m.FNO is calculated in __init__
        expected_max_freq = 1 / (m.resolved_wavelength * 1e-3 * m.FNO)
        assert abs(m.max_freq - expected_max_freq) < 1e-9, "max_freq 'cutoff' calculation mismatch."

    def test_max_freq_custom(self, optic_fixture, ensure_numpy_backend):
        """Test if max_freq is set correctly when a custom value is provided."""
        # be.set_backend("numpy")
        custom_freq = 100.0
        m = HuygensMTF(optic_fixture, num_rays=32, image_size=32, max_freq=custom_freq)
        assert m.max_freq == custom_freq, "Custom max_freq not set correctly."

    def test_freq_array_generation(self, optic_fixture, ensure_numpy_backend):
        """Test the generation of the frequency array."""
        # be.set_backend("numpy")
        image_s = 32
        m = HuygensMTF(optic_fixture, num_rays=32, image_size=image_s) # Ensure image_size is passed

        # freq = be.arange(image_size // 2) * df
        # df = 1.0 / (image_size * pixel_pitch_mm)
        assert hasattr(m, "freq"), "Frequency array 'freq' not generated."
        assert m.freq is not None, "Frequency array 'freq' is None."
        # Length can be up to image_size // 2, but might be truncated by max_freq
        assert len(m.freq) <= m.image_size // 2
        if len(m.freq) > 0:
            assert m.freq[0] == 0, "Frequency array should start at 0."

        # Check units calculation indirectly by ensuring freq values are reasonable
        # and that it doesn't exceed max_freq
        if len(m.freq) > 1:
             assert m.freq[-1] <= m.max_freq * (1 + 1e-9) # allow for small floating point inaccuracies if last point is exactly max_freq

        # More detailed check of _get_mtf_units if possible
        # Requires a known pixel_pitch.
        # psf_instance = m.psf_instances[0]
        # expected_df = 1.0 / (m.image_size * psf_instance.pixel_pitch)
        # if len(m.freq) > 1:
        #     assert abs(m.freq[1] - expected_df) < 1e-9 # Check first step

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
    def test_num_rays_and_grid_size(self, set_test_backend, num_rays, expected_pupil_sampling, optic):
        m = FFTMTF(optic, num_rays=num_rays, grid_size=None)

        assert m.num_rays == expected_pupil_sampling
        assert m.grid_size == 2 * num_rays
