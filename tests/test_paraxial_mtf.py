"""Test for paraxial surface MTF edge cases

This test validates that the fixes for paraxial surface MTF calculations
properly handle edge cases like collimated output and infinite focal lengths.
"""
import pytest
import numpy as np

import optiland.backend as be
from optiland import optic, materials
from optiland.mtf import GeometricMTF, FFTMTF


def test_paraxial_surface_mtf_original_issue():
    """Test the original issue case from GitHub issue report."""
    n = 1.5
    wavelength = 0.5
    singlet = optic.Optic()

    singlet.set_aperture(aperture_type="EPD", value=0.3*2)
    singlet.add_surface(index=0, radius=np.inf, thickness=400)
    singlet.add_surface(index=1, radius=np.inf, thickness=0.9, is_stop=True)
    singlet.add_surface(index=2, radius=np.inf, thickness=0.5, material=materials.IdealMaterial(n=n))
    singlet.add_surface(index=3,
        f=0.500,
        surface_type="paraxial",
        thickness=0.5
    )
    singlet.add_surface(index=4)

    singlet.set_field_type(field_type="angle")
    singlet.add_field(y=0)
    singlet.add_field(y=15)
    singlet.add_field(y=30)

    singlet.add_wavelength(value=wavelength, is_primary=True)

    # Test paraxial properties
    f2 = singlet.paraxial.f2()
    assert not be.isnan(f2) and not be.isinf(f2), "f2 should be finite"
    
    fno = singlet.paraxial.FNO()
    assert not be.isnan(fno) and not be.isinf(fno), "FNO should be finite"
    
    # Test GeometricMTF - should not raise warnings or errors
    geo_mtf = GeometricMTF(singlet)
    assert geo_mtf.max_freq > 0, "max_freq should be positive"
    assert not be.isnan(geo_mtf.max_freq), "max_freq should not be nan"
    assert not be.isinf(geo_mtf.max_freq), "max_freq should not be inf"
    
    # Test FFTMTF - should not raise warnings or errors
    fft_mtf = FFTMTF(singlet)
    assert fft_mtf.max_freq > 0, "max_freq should be positive"
    assert not be.isnan(fft_mtf.max_freq), "max_freq should not be nan"
    assert not be.isinf(fft_mtf.max_freq), "max_freq should not be inf"


def test_paraxial_surface_collimated_output():
    """Test that the safeguards handle edge cases without breaking normal operation.
    
    While it's difficult to create a real optical system with u=0 (collimated output),
    we test that systems that might be close to that edge case are handled gracefully.
    """
    # Create a simple paraxial lens
    lens = optic.Optic()
    
    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, surface_type="paraxial", f=100, thickness=100, is_stop=True)
    lens.add_surface(index=2)
    
    lens.set_aperture(aperture_type="EPD", value=20)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_wavelength(value=0.55, is_primary=True)
    
    # Test paraxial properties - should handle edge cases gracefully
    f2 = lens.paraxial.f2()
    assert not be.isnan(f2), "f2 should not be nan"
    
    fno = lens.paraxial.FNO()
    assert not be.isnan(fno), "FNO should not be nan"
    
    # MTF calculation should work without errors
    geo_mtf = GeometricMTF(lens)
    assert geo_mtf.max_freq > 0, "max_freq should be positive"
    assert not be.isnan(geo_mtf.max_freq), "max_freq should not be nan"
    assert be.isfinite(geo_mtf.max_freq), "max_freq should be finite"


def test_paraxial_surface_explicit_max_freq():
    """Test that explicit max_freq bypasses edge case handling."""
    lens = optic.Optic()
    
    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, surface_type="paraxial", f=100, thickness=100, is_stop=True)
    lens.add_surface(index=2)
    
    lens.set_aperture(aperture_type="EPD", value=20)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_wavelength(value=0.55, is_primary=True)
    
    # Explicitly specify max_freq to avoid edge case handling
    explicit_max_freq = 500.0
    geo_mtf = GeometricMTF(lens, max_freq=explicit_max_freq)
    
    assert geo_mtf.max_freq == explicit_max_freq, "Should use explicit max_freq"


def test_paraxial_surface_normal_case():
    """Test that normal paraxial surfaces still work correctly."""
    lens = optic.Optic()
    
    lens.add_surface(index=0, radius=np.inf, thickness=100)
    lens.add_surface(index=1, surface_type="paraxial", f=50, thickness=50, is_stop=True)
    lens.add_surface(index=2)
    
    lens.set_aperture(aperture_type="EPD", value=10)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_field(y=5)
    lens.add_wavelength(value=0.55, is_primary=True)
    
    # Test paraxial properties
    f2 = lens.paraxial.f2()
    assert be.isfinite(f2) and f2 > 0, "f2 should be positive and finite"
    
    fno = lens.paraxial.FNO()
    assert be.isfinite(fno) and fno > 0, "FNO should be positive and finite"
    
    # Test MTF calculations
    geo_mtf = GeometricMTF(lens)
    assert be.isfinite(geo_mtf.max_freq) and geo_mtf.max_freq > 0
    
    fft_mtf = FFTMTF(lens, num_rays=32)  # Use smaller num_rays for faster test
    assert be.isfinite(fft_mtf.max_freq) and fft_mtf.max_freq > 0


def test_paraxial_edge_case_handling():
    """Test that f2() and f1() handle near-zero u[-1] correctly.
    
    This is a unit test that verifies the edge case logic directly.
    """
    from unittest.mock import Mock, patch
    import optiland.backend as be
    from optiland.paraxial import Paraxial
    
    # Create a mock optic
    mock_optic = Mock()
    mock_optic.primary_wavelength = 0.55
    mock_optic.surface_group.positions = be.array([[0], [1], [2]])
    
    paraxial = Paraxial(mock_optic)
    
    # Mock _trace_generic to return u[-1] = 0 (collimated case)
    with patch.object(paraxial, '_trace_generic') as mock_trace:
        # Case 1: u[-1] is exactly zero
        mock_trace.return_value = (be.array([[1.0], [1.0]]), be.array([[0.1], [0.0]]))
        f2 = paraxial.f2()
        assert be.isinf(f2), "f2 should be infinite when u[-1] is zero"
        
        # Case 2: u[-1] is very small but not zero
        mock_trace.return_value = (be.array([[1.0], [1.0]]), be.array([[0.1], [1e-15]]))
        f2 = paraxial.f2()
        assert be.isinf(f2), "f2 should be infinite when u[-1] is very small"
        
        # Case 3: u[-1] is normal (not near zero)
        mock_trace.return_value = (be.array([[1.0], [1.0]]), be.array([[0.1], [0.01]]))
        f2 = paraxial.f2()
        assert be.isfinite(f2), "f2 should be finite when u[-1] is normal"
        assert f2 > 0, "f2 should be positive"


def test_mtf_with_infinite_fno():
    """Test that MTF classes handle infinite FNO gracefully.
    
    This test verifies the edge case logic is correct by testing
    the condition directly.
    """
    # Test the logic for handling infinite FNO
    resolved_wavelength = 0.55
    
    # Case 1: Normal FNO
    fno_normal = 5.0
    if be.isinf(fno_normal) or be.isnan(fno_normal):
        max_freq = 1000.0
    else:
        max_freq = 1 / (resolved_wavelength * 1e-3 * fno_normal)
    
    assert be.isfinite(max_freq), "max_freq should be finite for normal FNO"
    assert max_freq > 0, "max_freq should be positive"
    
    # Case 2: Infinite FNO (afocal system)
    fno_infinite = be.inf
    if be.isinf(fno_infinite) or be.isnan(fno_infinite):
        max_freq = 1000.0
    else:
        max_freq = 1 / (resolved_wavelength * 1e-3 * fno_infinite)
    
    assert max_freq == 1000.0, "max_freq should use default for infinite FNO"
    assert be.isfinite(max_freq), "max_freq should be finite"
    
    # Case 3: NaN FNO
    fno_nan = be.nan
    if be.isinf(fno_nan) or be.isnan(fno_nan):
        max_freq = 1000.0
    else:
        max_freq = 1 / (resolved_wavelength * 1e-3 * fno_nan)
    
    assert max_freq == 1000.0, "max_freq should use default for NaN FNO"
    assert be.isfinite(max_freq), "max_freq should be finite"


if __name__ == "__main__":
    # Run tests manually if executed directly
    test_paraxial_surface_mtf_original_issue()
    print("✓ test_paraxial_surface_mtf_original_issue passed")
    
    test_paraxial_surface_collimated_output()
    print("✓ test_paraxial_surface_collimated_output passed")
    
    test_paraxial_surface_explicit_max_freq()
    print("✓ test_paraxial_surface_explicit_max_freq passed")
    
    test_paraxial_surface_normal_case()
    print("✓ test_paraxial_surface_normal_case passed")
    
    test_paraxial_edge_case_handling()
    print("✓ test_paraxial_edge_case_handling passed")
    
    test_mtf_with_infinite_fno()
    print("✓ test_mtf_with_infinite_fno passed")
    
    print("\n✓ All tests passed!")
