import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from optiland.analysis import ThroughFocusMTF
from optiland.samples.objectives import CookeTriplet
import optiland.backend as be
import numpy as np


class TestThroughFocusMTF:
    """Test suite for the ThroughFocusMTF class."""

    def test_init_defaults(self, set_test_backend):
        """Test initialization with default parameters."""
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(optic, spatial_frequency=10.0)

        assert tfm.spatial_frequency == 10.0
        assert tfm.num_rays == 128
        assert tfm.wavelength == optic.primary_wavelength
        assert tfm.delta_focus == 0.1  # Default
        assert tfm.num_steps == 5  # Default
        assert len(tfm.fields) == len(optic.fields.fields)  # Default "all"
        assert tfm.wavelengths == [optic.primary_wavelength]

        # Check results structure (analysis is done in __init__ via parent)
        assert len(tfm.results) == tfm.num_steps
        assert len(tfm.results[0]) == len(tfm.fields)

    def test_init_custom_params(self, cooke_optic):
        """Test initialization with custom parameters."""
        custom_fields = [(0.0, 0.0), (0.0, 0.7)]
        custom_wl = 0.550
        tfm = ThroughFocusMTF(
            optic=cooke_optic,
            spatial_frequency=25.0,
            delta_focus=0.05,
            num_steps=3,
            fields=custom_fields,
            wavelength=custom_wl,
            num_rays=64,
        )

        assert tfm.spatial_frequency == 25.0
        assert tfm.num_rays == 64
        assert tfm.wavelength == custom_wl
        assert tfm.delta_focus == 0.05
        assert tfm.num_steps == 3
        assert tfm.fields == custom_fields
        assert tfm.wavelengths == [custom_wl]

        assert len(tfm.results) == 3
        assert len(tfm.results[0]) == len(custom_fields)

    def test_init_wavelength_direct_float(self, cooke_optic):
        """Test initialization with a specific float wavelength."""
        wavelength_val = 0.6328
        tfm = ThroughFocusMTF(cooke_optic, spatial_frequency=5.0, wavelength=wavelength_val)
        assert tfm.wavelength == wavelength_val
        assert tfm.wavelengths == [wavelength_val]

    def test_init_invalid_num_steps(self, cooke_optic):
        """Test initialization with invalid num_steps values."""
        # Test MIN_STEPS boundary
        with pytest.raises(ValueError, match="between 1 and 15"):
            ThroughFocusMTF(cooke_optic, 10.0, num_steps=0)

        # Test MAX_STEPS boundary
        with pytest.raises(ValueError, match="between 1 and 15"):
            ThroughFocusMTF(cooke_optic, 10.0, num_steps=17)

        # Test even number of steps
        with pytest.raises(ValueError, match="must be an odd integer"):
            ThroughFocusMTF(cooke_optic, 10.0, num_steps=4)

        # Test non-integer number of steps
        with pytest.raises(TypeError, match="must be an integer"):
            ThroughFocusMTF(cooke_optic, 10.0, num_steps=3.5)

    def test_init_optic_missing_primary_wavelength(self):
        """Test initialization with wavelength='primary' and optic missing attribute."""
        # Create a mock optic that doesn't have 'primary_wavelength'
        class MockOptic:
            pass
        mock_optic_instance = MockOptic()

        with pytest.raises(AttributeError, match="must have a 'primary_wavelength' attribute"):
            ThroughFocusMTF(mock_optic_instance, spatial_frequency=10.0, wavelength="primary")

    def test_analysis_results_structure_and_values(self, default_tf_mtf):
        """Test the structure and basic validity of the analysis results."""
        tfm = default_tf_mtf
        assert len(tfm.results) == tfm.num_steps

        for step_result in tfm.results:
            assert len(step_result) == len(tfm.fields)
            for field_result in step_result:
                assert "tangential" in field_result
                assert "sagittal" in field_result
                # MTF values should be between 0 and 1 (inclusive)
                assert 0.0 <= field_result["tangential"] <= 1.0
                assert 0.0 <= field_result["sagittal"] <= 1.0
                # Check they are numpy floats (or compatible) after be.to_numpy
                assert isinstance(field_result["tangential"], (float, np.floating))
                assert isinstance(field_result["sagittal"], (float, np.floating))


    @patch("matplotlib.pyplot.show")
    def test_view_min_steps(self, mock_show, cooke_optic):
        """Test view method with num_steps=1 (k=0 spline order)."""
        tfm = ThroughFocusMTF(cooke_optic, spatial_frequency=10.0, num_steps=1)
        tfm.view()
        mock_show.assert_called_once()
        plt.close()  # Close the figure to free resources

    @patch("matplotlib.pyplot.show")
    def test_view_few_steps(self, mock_show, cooke_optic):
        """Test view method with num_steps=3 (k=1 spline order)."""
        tfm = ThroughFocusMTF(cooke_optic, spatial_frequency=10.0, num_steps=3)
        tfm.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_default_steps(self, mock_show, default_tf_mtf):
        """Test view method with default num_steps=5 (k=3 spline order)."""
        default_tf_mtf.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_single_field(self, mock_show, cooke_optic):
        """Test view method with a single specified field."""
        tfm = ThroughFocusMTF(
            cooke_optic,
            spatial_frequency=10.0,
            num_steps=5,
            fields=[(0.0, 0.0)] # Single field
        )
        assert len(tfm.fields) == 1
        tfm.view()
        mock_show.assert_called_once()
        # To check plot content more deeply, one might inspect the ax object
        # For example, check number of lines:
        # fig = plt.gcf()
        # ax = fig.axes[0]
        # assert len(ax.lines) == 2 # One tangential, one sagittal for the single field
        plt.close()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots") # Patch subplots to check figsize
    def test_view_custom_figsize(self, mock_subplots, mock_show, default_tf_mtf):
        """Test view method with a custom figsize."""
        # Create mock figure and axes objects to be returned by subplots
        mock_fig = MagicMock(spec=plt.Figure)
        mock_ax = MagicMock(spec=plt.Axes)
        mock_subplots.return_value = (mock_fig, mock_ax)

        custom_figsize = (10, 5)
        default_tf_mtf.view(figsize=custom_figsize)

        mock_subplots.assert_called_once_with(figsize=custom_figsize)
        mock_show.assert_called_once()
        plt.close() # Ensure plot is closed even if it's mocked

    @patch("matplotlib.pyplot.show")
    def test_view_all_fields_from_optic(self, mock_show, cooke_optic):
        """Test view method when using all fields from the optic."""
        tfm = ThroughFocusMTF(cooke_optic, spatial_frequency=10.0, fields="all")
        assert len(tfm.fields) == len(cooke_optic.fields) # Ensure "all" was processed
        tfm.view()
        mock_show.assert_called_once()
        plt.close()
