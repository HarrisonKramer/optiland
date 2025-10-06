# tests/paraxial/test_paraxial_to_thick.py
"""
Tests for the `paraxial_to_thick_lens` utility function in
optiland.paraxial.paraxial_to_thick.
"""
import pytest

import optiland.backend as be
from optiland.surfaces import convert_to_thick_lens as paraxial_to_thick_lens
from optiland.optic import Optic
from ..utils import assert_allclose


@pytest.fixture
def paraxial_optic():
    """
    Provides a simple optical system containing a single paraxial lens surface
    for conversion testing.
    """
    lens = Optic()
    lens.add_surface(index=0, thickness=be.inf)
    lens.add_surface(
        index=1,
        surface_type="paraxial",
        thickness=100,
        f=100,
        is_stop=True,
        material="N-BK7",
    )
    lens.add_surface(index=2)
    lens.set_aperture(aperture_type="EPD", value=20)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_wavelength(value=0.55, is_primary=True)
    return lens


class TestParaxialToThick:
    """
    Tests the `paraxial_to_thick_lens` function, which converts an idealized
    paraxial lens into a physically realistic thick lens with two surfaces.
    """

    def test_equiconvex_conversion(self, set_test_backend, paraxial_optic):
        """
        Tests the conversion to an equiconvex lens, where both surfaces have
        the same radius of curvature (with opposite signs).
        """
        paraxial_to_thick_lens(paraxial_optic, 1, thickness=5.0, bend=0.0)
        # Check that the number of surfaces has increased by one
        assert paraxial_optic.surface_group.num_surfaces == 4
        # Check that the new surfaces have the correct radii
        assert_allclose(paraxial_optic.surface_group.radii[1], 103.36, atol=1e-2)
        assert_allclose(paraxial_optic.surface_group.radii[2], -103.36, atol=1e-2)
        # Verify that the focal length of the new thick lens is correct
        assert_allclose(paraxial_optic.paraxial.f2(), 100.0, atol=1e-2)

    def test_plano_convex_conversion(self, set_test_backend, paraxial_optic):
        """
        Tests the conversion to a plano-convex lens, where one surface is flat
        and the other is curved.
        """
        paraxial_to_thick_lens(paraxial_optic, 1, thickness=5.0, bend=1.0)
        assert paraxial_optic.surface_group.num_surfaces == 4
        assert_allclose(paraxial_optic.surface_group.radii[1], 51.68, atol=1e-2)
        assert paraxial_optic.surface_group.radii[2] == be.inf
        assert_allclose(paraxial_optic.paraxial.f2(), 100.0, atol=1e-2)

    def test_convex_plano_conversion(self, set_test_backend, paraxial_optic):
        """
        Tests the conversion to a convex-plano lens (the reverse of
        plano-convex).
        """
        paraxial_to_thick_lens(paraxial_optic, 1, thickness=5.0, bend=-1.0)
        assert paraxial_optic.surface_group.num_surfaces == 4
        assert paraxial_optic.surface_group.radii[1] == be.inf
        assert_allclose(paraxial_optic.surface_group.radii[2], -51.68, atol=1e-2)
        assert_allclose(paraxial_optic.paraxial.f2(), 100.0, atol=1e-2)

    def test_custom_bend_conversion(self, set_test_backend, paraxial_optic):
        """
        Tests the conversion with a custom bending factor, resulting in a
        biconvex lens with unequal radii.
        """
        paraxial_to_thick_lens(paraxial_optic, 1, thickness=5.0, bend=0.5)
        assert paraxial_optic.surface_group.num_surfaces == 4
        assert_allclose(paraxial_optic.surface_group.radii[1], 68.90, atol=1e-2)
        assert_allclose(paraxial_optic.surface_group.radii[2], -206.71, atol=1e-2)
        assert_allclose(paraxial_optic.paraxial.f2(), 100.0, atol=1e-2)

    def test_invalid_surface_type_error(self, set_test_backend, paraxial_optic):
        """
        Tests that attempting to convert a non-paraxial surface raises a
        ValueError.
        """
        # Change the surface type to 'standard' to make it invalid for conversion
        paraxial_optic.surface_group.surfaces[1].surface_type = "standard"
        with pytest.raises(ValueError, match="Surface is not a paraxial surface."):
            paraxial_to_thick_lens(paraxial_optic, 1)