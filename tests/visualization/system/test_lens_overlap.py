import warnings
import pytest
import numpy as np
from optiland.visualization.system.lens import Lens2D
from optiland.visualization.system.surface import Surface2D
from optiland.samples.objectives import CookeTriplet

def test_lens_overlap_warning():
    """Test that Lens2D raises a warning when surfaces overlap if check_overlap is True."""
    optic = CookeTriplet()

    # Access surfaces and create overlap
    s1 = optic.surface_group.surfaces[1]
    s2 = optic.surface_group.surfaces[2]

    # Create overlap by setting negative thickness
    # Must use optic.set_thickness to update global coordinates
    optic.set_thickness(-5.0, 1)

    # Create wrappers
    s1_2d = Surface2D(s1, ray_extent=10.0)
    s2_2d = Surface2D(s2, ray_extent=10.0)

    # Check warning
    with pytest.warns(UserWarning, match="overlap detected"):
        Lens2D([s1_2d, s2_2d], check_overlap=True)

def test_lens_overlap_no_warning():
    """Test that Lens2D does not raise a warning when surfaces overlap if check_overlap is False."""
    optic = CookeTriplet()

    # Access surfaces and create overlap
    s1 = optic.surface_group.surfaces[1]
    s2 = optic.surface_group.surfaces[2]

    optic.set_thickness(-5.0, 1)

    s1_2d = Surface2D(s1, ray_extent=10.0)
    s2_2d = Surface2D(s2, ray_extent=10.0)

    # Check no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Lens2D([s1_2d, s2_2d], check_overlap=False)
        # Filter out unrelated warnings if any, but we expect none related to overlap
        overlap_warnings = [x for x in w if "overlap detected" in str(x.message)]
        assert len(overlap_warnings) == 0

def test_lens_no_overlap_warning():
    """Test that Lens2D does not raise a warning when surfaces do not overlap."""
    optic = CookeTriplet()

    s1 = optic.surface_group.surfaces[1]
    s2 = optic.surface_group.surfaces[2]

    # Standard CookeTriplet has no overlaps

    s1_2d = Surface2D(s1, ray_extent=10.0)
    s2_2d = Surface2D(s2, ray_extent=10.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Lens2D([s1_2d, s2_2d], check_overlap=True)
        overlap_warnings = [x for x in w if "overlap detected" in str(x.message)]
        assert len(overlap_warnings) == 0
