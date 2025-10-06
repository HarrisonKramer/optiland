# tests/mtf/test_sampled_mtf.py
"""
Tests for the SampledMTF class in optiland.mtf.
"""
from optiland.mtf import SampledMTF
from optiland.samples.objectives import CookeTriplet
from ..utils import assert_allclose


class TestSampledMTF:
    """
    Tests the SampledMTF class, which calculates the MTF from Zernike
    coefficients fitted to a sampled wavefront.
    """

    def test_sampled_mtf_instantiation(self, set_test_backend):
        """
        Tests that the SampledMTF class can be instantiated correctly with
        various parameters.
        """
        optic = CookeTriplet()
        field = (0, 0)
        wavelength = optic.primary_wavelength

        # Test instantiation with default and custom parameters
        sampled_mtf_instance = SampledMTF(
            optic=optic,
            field=field,
            wavelength=wavelength,
            num_rays=32,
            distribution="uniform",
            zernike_terms=37,
            zernike_type="fringe",
        )
        assert sampled_mtf_instance is not None
        assert sampled_mtf_instance.optic == optic
        assert sampled_mtf_instance.num_rays == 32

    def test_mtf_at_zero_frequency(self, set_test_backend):
        """
        Tests that the MTF at zero spatial frequency is always 1.0, as per
        the definition of MTF.
        """
        optic = CookeTriplet()
        field = optic.fields.get_field_coords()[0]
        sampled_mtf_instance = SampledMTF(optic=optic, field=field, wavelength=0.55)

        frequencies = [(0.0, 0.0)]
        mtf_values = sampled_mtf_instance.calculate_mtf(frequencies)

        assert len(mtf_values) == 1
        assert_allclose(mtf_values[0], 1.0)

    def test_triplet_system_behavior(self, set_test_backend):
        """
        Tests the MTF calculation for a well-behaved system (Cooke Triplet)
        against known reference values at several spatial frequencies.
        """
        optic = CookeTriplet()
        field = (0, 0)
        sampled_mtf_instance = SampledMTF(
            optic=optic, field=field, wavelength=0.55, num_rays=32, zernike_terms=37
        )

        freqs = [(30.0, 0.0), (0.0, 30.0), (20.0, 20.0)]
        mtf_values = sampled_mtf_instance.calculate_mtf(frequencies=freqs)

        assert len(mtf_values) == len(freqs)
        assert_allclose(mtf_values, [0.77596, 0.77596, 0.79466], atol=1e-5)

    def test_defocused_system_behavior(self, set_test_backend):
        """
        Tests that introducing defocus into the system correctly lowers the
        MTF response, as expected.
        """
        optic = CookeTriplet()
        field = (0, 1)
        # Introduce defocus by shifting the image surface
        optic.image_surface.geometry.cs.z += 0.5
        sampled_mtf_defocused = SampledMTF(optic=optic, field=field, wavelength=0.6)

        freq = (25.0, 0.0)
        mtf_values = sampled_mtf_defocused.calculate_mtf(frequencies=[freq])
        assert_allclose(mtf_values, 0.044568, atol=1e-5)

    def test_calculate_mtf_multiple_calls(self, set_test_backend):
        """
        Tests that calling `calculate_mtf` multiple times for different
        frequencies yields the same results as a single combined call.
        """
        optic = CookeTriplet()
        field = (0.5, 0.5)
        sampled_mtf_instance = SampledMTF(optic=optic, field=field, wavelength=0.6)

        freq1 = (5.0, 0.0)
        freq2 = (10.0, 0.0)
        mtf_val1_list = sampled_mtf_instance.calculate_mtf(frequencies=[freq1])
        mtf_val2_list = sampled_mtf_instance.calculate_mtf(frequencies=[freq2])
        mtf_vals_combined_list = sampled_mtf_instance.calculate_mtf(frequencies=[freq1, freq2])

        assert_allclose(mtf_val1_list + mtf_val2_list, mtf_vals_combined_list)

    def test_zero_xpd_handling(self, set_test_backend):
        """
        Tests that if the exit pupil diameter (XPD) is zero (e.g., vignetted),
        the MTF for any non-zero frequency is correctly calculated as zero.
        """
        optic = CookeTriplet()
        field = (0, 0)
        sampled_mtf_instance = SampledMTF(optic=optic, field=field, wavelength=0.56)
        # Manually set XPD to zero to simulate vignetting
        sampled_mtf_instance.xpd = 0.0

        frequencies = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (5.0, 5.0)]
        expected_mtfs = [1.0, 0.0, 0.0, 0.0]
        mtf_values = sampled_mtf_instance.calculate_mtf(frequencies=frequencies)

        assert_allclose(mtf_values, expected_mtfs)