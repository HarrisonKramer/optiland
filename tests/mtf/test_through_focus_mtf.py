# tests/mtf/test_through_focus_mtf.py
"""
Tests for the ThroughFocusMTF analysis tool in optiland.mtf.
"""
import pytest
import matplotlib.pyplot as plt

from optiland.analysis import ThroughFocusMTF
from optiland.samples.objectives import CookeTriplet


class TestThroughFocusMTF:
    """
    Tests the ThroughFocusMTF class, which calculates MTF at a specific
    spatial frequency across a range of focal positions.
    """

    def test_init_defaults(self, set_test_backend):
        """
        Tests that the class initializes correctly with default parameters.
        """
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(optic, spatial_frequency=10.0)
        assert tfm.spatial_frequency == 10.0
        assert tfm.num_rays == 128
        assert tfm.wavelength == optic.primary_wavelength
        assert tfm.delta_focus == 0.1
        assert tfm.num_steps == 5
        assert len(tfm.fields) == len(optic.fields.fields)

    def test_init_custom_params(self, set_test_backend):
        """
        Tests that the class initializes correctly with custom parameters for
        all settings.
        """
        optic = CookeTriplet()
        custom_fields = [(0.0, 0.0), (0.1, 0.7)]
        custom_wl = 0.550
        tfm = ThroughFocusMTF(
            optic=optic,
            spatial_frequency=25.0,
            delta_focus=0.05,
            num_steps=3,
            fields=custom_fields,
            wavelength=custom_wl,
            num_rays=23,
        )
        assert tfm.spatial_frequency == 25.0
        assert tfm.num_rays == 23
        assert tfm.wavelength == custom_wl
        assert tfm.delta_focus == 0.05
        assert tfm.num_steps == 3
        assert tfm.fields == custom_fields

    def test_init_invalid_num_steps(self, set_test_backend):
        """
        Tests that initialization raises a ValueError for an invalid number
        of steps (must be an odd integer between 1 and 15).
        """
        optic = CookeTriplet()
        with pytest.raises(ValueError):
            ThroughFocusMTF(optic, 10.0, num_steps=0)  # Too small
        with pytest.raises(ValueError):
            ThroughFocusMTF(optic, 10.0, num_steps=17)  # Too large
        with pytest.raises(ValueError):
            ThroughFocusMTF(optic, 10.0, num_steps=4)  # Even number
        with pytest.raises(ValueError):
            ThroughFocusMTF(optic, 10.0, num_steps=3.5)  # Not an integer

    def test_analysis_results_structure_and_values(self, set_test_backend):
        """
        Tests the structure of the generated results and ensures that all MTF
        values are physically plausible (between 0 and 1).
        """
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(optic, spatial_frequency=25.0, num_steps=1, num_rays=23)
        assert len(tfm.results) == tfm.num_steps

        for step_result in tfm.results:
            assert len(step_result) == len(tfm.fields)
            for field_result in step_result:
                assert "tangential" in field_result
                assert "sagittal" in field_result
                assert 0.0 <= field_result["tangential"] <= 1.0
                assert 0.0 <= field_result["sagittal"] <= 1.0

    def test_view_with_spline_orders(self, set_test_backend):
        """
        Tests the view method with different `num_steps` to ensure it handles
        various spline interpolation orders (k) correctly.
        """
        optic = CookeTriplet()
        # Test with num_steps=1 (k=0 spline)
        tfm1 = ThroughFocusMTF(optic, spatial_frequency=10.0, num_steps=1)
        fig1, ax1 = tfm1.view()
        assert fig1 is not None and ax1 is not None
        plt.close(fig1)

        # Test with num_steps=3 (k=1 spline)
        tfm3 = ThroughFocusMTF(optic, spatial_frequency=10.0, num_steps=3)
        fig3, ax3 = tfm3.view()
        assert fig3 is not None and ax3 is not None
        plt.close(fig3)

        # Test with num_steps=5 (k=3 spline)
        tfm5 = ThroughFocusMTF(optic, spatial_frequency=152.0, num_rays=53)
        fig5, ax5 = tfm5.view()
        assert fig5 is not None and ax5 is not None
        plt.close(fig5)

    def test_view_single_field(self, set_test_backend):
        """
        Tests that the view method correctly generates a plot when only a
        single field is specified.
        """
        optic = CookeTriplet()
        tfm = ThroughFocusMTF(optic, spatial_frequency=10.0, fields=[(0.0, 0.0)])
        assert len(tfm.fields) == 1
        fig, ax = tfm.view()
        assert fig is not None and ax is not None
        plt.close(fig)