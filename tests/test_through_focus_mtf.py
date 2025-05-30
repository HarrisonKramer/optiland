import pytest
from unittest.mock import patch, MagicMock, call # Added 'call' for more detailed mock assertions later

import numpy as np # For test data generation
import optiland.backend as be # For backend consistency if needed in tests

# Import the class to be tested
from optiland.analysis.through_focus_mtf import ThroughFocusMTF

# Imports needed for creating a basic Optic instance for testing
from optiland.optic import Optic
from optiland.surfaces import ObjectSurface, ImageSurface, StandardSurface
from optiland.geometries import Plane as PlaneGeometry # Aliased to avoid conflict with other 'Plane'
from optiland.fields import Fields
from optiland.wavelength import Wavelengths
from optiland.materials import IdealMaterial # For basic surfaces

# Import SampledMTF for mocking
from optiland.mtf import SampledMTF

# Import ThroughFocusAnalysis to potentially mock its __init__ if needed for specific tests
from optiland.analysis.through_focus import ThroughFocusAnalysis


@pytest.fixture
def basic_optic():
    """Provides a simple Optic instance for testing."""

    # 1. Wavelengths
    wavelengths = Wavelengths([0.550], primary_wavelength_index=0) # 0.550 µm = 550 nm

    # 2. Fields
    fields = Fields(field_points=[(0.0, 0.0), (0.0, 1.0)]) # On-axis and one off-axis field (y=1mm)

    # 3. Surfaces
    # Object surface (at infinity)
    object_surface = ObjectSurface(is_infinite=True)

    # Surface 1: Plane surface acting as the stop
    # Made of IdealMaterial (e.g., glass n=1.5)
    # Thickness is the distance to the next surface (or image plane if last surface)
    surface1 = StandardSurface(
        geometry=PlaneGeometry(),
        material=IdealMaterial(1.5),
        thickness=50.0, # Distance from this surface to the image plane
        semi_diameter=20.0,
        is_stop=True # This surface is the aperture stop
    )

    # Image surface (where the image is formed)
    image_surface = ImageSurface()

    # Create the Optic instance
    # The `surfaces` list contains all surfaces between object and image.
    # `entrance_pupil_radius` needs to be set for systems where it cannot be automatically determined
    # or to override automatic determination. For a system with a defined stop surface,
    # the EPD is determined by that stop's semi-diameter if object is at infinity.
    # However, explicit setting as per instructions.
    optic = Optic(
        object_surface=object_surface,
        image_surface=image_surface,
        surfaces=[surface1], # Only surface1 is between object and image
        wavelengths=wavelengths,
        fields=fields,
        entrance_pupil_radius=5.0 # Explicit EPD of 5mm radius (10mm diameter)
    )

    # Ensure paraxial properties are available.
    # Optic class usually computes these on demand or during initialization.
    # Accessing them here can help catch setup issues early in tests that use this fixture.
    # For example, if optic.paraxial.XPD() or optic.paraxial.XPL() raised an error,
    # it would indicate a problem with the optic's setup for MTF calculations.
    # No explicit call to update() is typically needed unless specified by Optic's usage.
    # print(f"Fixture XPD: {optic.paraxial.XPD()}, XPL: {optic.paraxial.XPL()}") # For debugging if needed

    return optic


# Tests for __init__ method

@patch('optiland.analysis.through_focus_mtf.ThroughFocusAnalysis.__init__', return_value=None)
def test_tf_mtf_init_basic(mock_super_init, basic_optic):
    """Tests basic attribute storage during __init__."""
    tf_mtf = ThroughFocusMTF(
        optic=basic_optic,
        spatial_frequency=10.0,
        num_rays=32
    )
    assert tf_mtf.spatial_frequency == 10.0
    assert tf_mtf.num_rays == 32
    assert tf_mtf.optic is basic_optic
    # Check that super().__init__ was called (even if just to confirm setup for other tests)
    mock_super_init.assert_called_once()


@patch('optiland.analysis.through_focus_mtf.ThroughFocusAnalysis.__init__', return_value=None)
def test_tf_mtf_init_wavelength_resolution(mock_super_init, basic_optic):
    """Tests how wavelength is resolved during __init__."""
    # Test primary wavelength
    tf_mtf_primary = ThroughFocusMTF(
        optic=basic_optic,
        spatial_frequency=5.0,
        wavelength="primary"
    )
    assert tf_mtf_primary.wavelength == basic_optic.primary_wavelength

    # Test specific wavelength
    tf_mtf_specific = ThroughFocusMTF(
        optic=basic_optic,
        spatial_frequency=5.0,
        wavelength=0.632
    )
    assert tf_mtf_specific.wavelength == 0.632
    # mock_super_init will be called twice here across the two instantiations
    # For this test, we only care about tf_mtf_primary.wavelength and tf_mtf_specific.wavelength


@patch('optiland.analysis.through_focus_mtf.ThroughFocusAnalysis.__init__', return_value=None)
def test_tf_mtf_init_super_call(mock_super_init, basic_optic):
    """Tests that the superclass __init__ is called correctly."""
    tf_mtf = ThroughFocusMTF(
        optic=basic_optic,
        spatial_frequency=10.0,
        delta_focus=0.05,
        num_steps=3,
        fields=[(0, 0)],
        wavelength=0.500, # Specific wavelength
        num_rays=16
    )

    mock_super_init.assert_called_once()
    args, kwargs = mock_super_init.call_args
    assert kwargs['optic'] is basic_optic
    assert kwargs['delta_focus'] == 0.05
    assert kwargs['num_steps'] == 3
    assert kwargs['fields'] == [(0, 0)]
    assert kwargs['wavelengths'] == [0.500] # Ensure it's a list with the resolved wavelength

    # Test with "primary" wavelength
    mock_super_init.reset_mock()
    tf_mtf_primary = ThroughFocusMTF(
        optic=basic_optic,
        spatial_frequency=12.0,
        delta_focus=0.02,
        num_steps=5,
        fields="all", # Test "all" fields string
        wavelength="primary",
        num_rays=32
    )
    mock_super_init.assert_called_once()
    args, kwargs = mock_super_init.call_args
    assert kwargs['optic'] is basic_optic
    assert kwargs['delta_focus'] == 0.02
    assert kwargs['num_steps'] == 5
    assert kwargs['fields'] == "all"
    assert kwargs['wavelengths'] == [basic_optic.primary_wavelength]


@pytest.mark.parametrize("invalid_step", [0, 2, 8, -1, 1.5]) # Added non-integer
def test_tf_mtf_init_num_steps_validation_invalid(basic_optic, invalid_step):
    """Tests that __init__ raises ValueError for invalid num_steps."""
    # This validation happens in ThroughFocusAnalysis.__init__ via _validate_num_steps
    # So, we DO NOT mock super().__init__ here.
    # We DO need to mock SampledMTF if a valid num_steps case would call it.
    with pytest.raises(ValueError):
        ThroughFocusMTF(
            optic=basic_optic,
            spatial_frequency=10.0,
            num_steps=invalid_step
        )

@patch('optiland.analysis.through_focus_mtf.SampledMTF') # Mock SampledMTF to prevent actual calculation
def test_tf_mtf_init_num_steps_validation_valid(mock_sampled_mtf, basic_optic):
    """Tests that __init__ does NOT raise ValueError for valid num_steps."""
    # This validation happens in ThroughFocusAnalysis.__init__
    # We mock SampledMTF because if num_steps is valid, super().__init__() will
    # proceed to call _perform_analysis_at_focus via run_analysis and _calculate_through_focus.
    try:
        ThroughFocusMTF(
            optic=basic_optic,
            spatial_frequency=10.0,
            num_steps=3 # A valid odd number
        )
        ThroughFocusMTF(
            optic=basic_optic,
            spatial_frequency=10.0,
            num_steps=5 # Another valid odd number
        )
        ThroughFocusMTF(
            optic=basic_optic,
            spatial_frequency=10.0,
            num_steps=1 # Smallest valid odd number
        )
        ThroughFocusMTF(
            optic=basic_optic,
            spatial_frequency=10.0,
            num_steps=7 # Max default valid odd number
        )
    except ValueError:
        pytest.fail("ThroughFocusMTF raised ValueError unexpectedly for valid num_steps")

    # We can also assert that SampledMTF was called if needed,
    # e.g., mock_sampled_mtf.assert_called()
    # For num_steps=3, _perform_analysis_at_focus is called 3 times.
    # Each time it iterates over fields (2 fields in basic_optic).
    # So SampledMTF constructor should be called 3 * 2 = 6 times for the first instance.
    # This check is more for the _perform_analysis_at_focus logic though.
    # For now, just ensuring no error is sufficient for this __init__ test.
    assert mock_sampled_mtf.called # At least it was called due to super().run_analysis()
