import pytest
import numpy as np # Using numpy for test assertions, be might be optiland's backend
from optiland.optic import Optic
from optiland.surfaces.paraxial_surface import ParaxialSurface, ParaxialToThickLensConverter
from optiland.surfaces.standard_surface import Surface
from optiland.materials.material import Material
from optiland.materials.ideal import IdealMaterial
from optiland.geometries.standard import StandardGeometry # For checking created surface geometry
import optiland.backend as be # For be.inf

# Helper for comparing float values, similar to tests/utils.py if it exists
# For now, defining a simple one here.
def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    assert np.allclose(a, b, rtol=rtol, atol=atol), f"Assertion failed: {a} vs {b}"

# Test cases will be added here in later steps.

# Example fixture structure (can be expanded)
@pytest.fixture
def basic_paraxial_setup():
    """Provides a basic Optic with a single ParaxialSurface."""
    # Re-doing optic setup for clarity in fixture
    optic_fixture = Optic(name="TestOpticForConversion")
    optic_fixture.add_surface(index=0, thickness=be.inf, comment="Object") # Object surface at infinity

    # Add the paraxial surface that will be converted
    # The surface_factory within SurfaceGroup is used by optic.add_surface
    # ParaxialSurface.__init__ expects focal_length, geometry, material_pre, material_post
    # Optic.add_surface simplifies this.
    optic_fixture.add_surface(
        index=1,
        surface_type="paraxial",
        f=100.0,
        thickness=50.0, # This is the space *after* the paraxial lens
        comment="Paraxial to Convert"
    )
    optic_fixture.add_surface(index=2, comment="Image") # Image surface

    optic_fixture.add_wavelength(value=0.55, is_primary=True) # Needed for n lookup

    # The paraxial surface to be converted is at index 1
    paraxial_surface_to_convert = optic_fixture.surface_group.surfaces[1]

    return optic_fixture, paraxial_surface_to_convert


def test_default_conversion(basic_paraxial_setup):
    """
    Tests default conversion using N-BK7 material and biconvex shape.
    Verifies surface count, removal of paraxial, type of new surfaces,
    material, radii, and center thickness.
    """
    optic, paraxial_surf_to_convert = basic_paraxial_setup
    original_num_surfaces = len(optic.surface_group.surfaces)
    paraxial_surf_original_idx = optic.surface_group.surfaces.index(paraxial_surf_to_convert)

    # Store properties of the original paraxial surface for later comparison/use
    original_focal_length = paraxial_surf_to_convert.f
    # original_z_position = paraxial_surf_to_convert.geometry.cs.z # For alignment checks later

    converter = ParaxialToThickLensConverter(
        paraxial_surface=paraxial_surf_to_convert,
        optic=optic,
        # Default material "N-BK7"
        # Default center_thickness 3.0
        # Default lens_shape "biconvex"
    )

    new_s1, new_s2 = converter.convert()

    # 1. Assert ParaxialSurface is removed and new surfaces are Standard Surfaces
    assert paraxial_surf_to_convert not in optic.surface_group.surfaces
    assert isinstance(optic.surface_group.surfaces[paraxial_surf_original_idx], Surface)
    assert not isinstance(optic.surface_group.surfaces[paraxial_surf_original_idx], ParaxialSurface)
    assert isinstance(optic.surface_group.surfaces[paraxial_surf_original_idx + 1], Surface)
    assert not isinstance(optic.surface_group.surfaces[paraxial_surf_original_idx + 1], ParaxialSurface)

    # 2. Assert correct number of surfaces
    # Optic started with Object, Paraxial, Image = 3 surfaces
    # Should now be Object, NewS1, NewS2, Image = 4 surfaces
    assert len(optic.surface_group.surfaces) == original_num_surfaces + 1

    # 3. Verify created surfaces are the ones returned and are at correct positions
    assert optic.surface_group.surfaces[paraxial_surf_original_idx] is new_s1
    assert optic.surface_group.surfaces[paraxial_surf_original_idx + 1] is new_s2

    # 4. Verify material of the new lens
    assert isinstance(new_s1.material_post, Material)
    assert new_s1.material_post.name.upper() == "N-BK7" # Material name might have case variations
    assert new_s2.material_pre is new_s1.material_post # Should be the same material instance

    # Check surrounding materials (should be air or original)
    # Assuming default 'air' for material_pre of first optical surface and material_post of last optical surface
    # The fixture sets up object at inf (air after), paraxial (air before/after by default), image (air before)
    # original_paraxial_surf_pre_material = paraxial_surf_to_convert.material_pre
    # original_paraxial_surf_post_material = paraxial_surf_to_convert.material_post
    # assert new_s1.material_pre is original_paraxial_surf_pre_material # This might change if Optic re-assigns default air
    # assert new_s2.material_post is original_paraxial_surf_post_material

    # A simpler check for now: ensure they are IdealMaterial if they were air
    # This depends on how Optic handles default materials when surfaces are added/removed.
    # ParaxialSurface default materials are IdealMaterial(n=1.0) for air.
    assert isinstance(new_s1.material_pre, IdealMaterial)
    assert_allclose(new_s1.material_pre.n(0.55), 1.0)
    assert isinstance(new_s2.material_post, IdealMaterial)
    assert_allclose(new_s2.material_post.n(0.55), 1.0)


    # 5. Verify center thickness
    # Thickness of surface1 is the center thickness of the lens
    assert_allclose(new_s1.thickness, 3.0) # Default center_thickness

    # 6. Verify radii are plausible for biconvex
    # For f=100, N-BK7 (n~1.5168 at 0.587um, let's use primary 0.55um, n~1.5185), d=3.0
    # R1 = (n-1) * 2 * f_effective (for thin lens approx, R1=-R2)
    # R1_approx = (1.5185-1) * 2 * 100 = 0.5185 * 200 = 103.7
    # For thick lens, it will be different.
    # P*n*R1^2 - 2*n*(n-1)*R1 + (n-1)^2*d = 0
    # P = 1/100 = 0.01
    # n_val = new_s1.material_post.n(optic.primary_wavelength) # approx 1.5185 for N-BK7 at 0.55um
    # Expected R1 for biconvex f=100, d=3, n=1.518508 (N-BK7 @ 0.55um from MaterialFile)
    # R1 should be around 102.3 mm for symmetric biconvex.
    assert new_s1.geometry.radius > 0
    assert new_s2.geometry.radius < 0
    # A more precise check would require calculating the expected R from the formula
    # For now, just check signs and rough symmetry if applicable.
    # With default biconvex, R1 should be -R2
    assert_allclose(new_s1.geometry.radius, -new_s2.geometry.radius, rtol=1e-2) # Allow some tolerance for calculation precision

    # 7. Check focal length preservation (paraxial behavior)
    # This is a more involved test, requiring tracing or paraxial analysis.
    # For now, we trust the _calculate_radii aims for this.
    # Add a specific test for this later.
    # Paraxial focal length of the system:
    # Optic().paraxial.f2() gives system focal length
    # We expect the new system with the thick lens to have a similar focal length
    # to a system where the paraxial lens was ideal.
    # This is tricky because the original paraxial surface *is* the system for EFL.
    # The EFL of the new Optic should be close to original_focal_length.

    # Update optic to recalculate paraxial properties
    # optic.update() is called by converter.convert()
    system_focal_length = optic.paraxial.f2() # Effective focal length in image space
    assert_allclose(system_focal_length, original_focal_length, rtol=0.05) # Allow 5% tolerance initially

    # 8. Check geometric center alignment
    # Original paraxial surface z position (assuming it was not object/image)
    # In basic_paraxial_setup, object is at inf (z=-inf effectively for rays, vertex at 0)
    # Paraxial surface (index 1) is placed at z=0 by Optic's default for first optical surface.
    # Expected S1 vertex z = original_paraxial_z - center_thickness / 2
    # Expected S2 vertex z = original_paraxial_z + center_thickness / 2
    # This check depends on the original_z_position and how _insert_new_surfaces works.

    # The fixture places the first optical element (paraxial surf) at z=0.
    # So, original_paraxial_z_pos = 0.
    # desired_s1_z = 0 - 3.0/2 = -1.5
    # desired_s2_z = 0 + 3.0/2 = 1.5 (relative to original paraxial z)
    # After insertion and optic.update(), surface_before_s1.thickness is adjusted.
    # If surface_before_s1 is object@inf, s1_z will be 0.
    # In this case, the center alignment logic might place the lens centered at z=0,
    # meaning s1_z = -1.5, s2_z = 1.5.
    # However, Optic() conventions usually place the first optical surface vertex at z=0
    # if it follows an object at infinity.
    # The _insert_new_surfaces tries to adjust thickness of surface before s1.
    # If object is at inf, this means obj.thickness (dist to first element) should be -1.5
    # But obj.thickness is usually positive or inf.
    # Let's check actual positions:
    s1_z = new_s1.geometry.cs.z
    s2_z = new_s2.geometry.cs.z

    # In basic_paraxial_setup, object surface is at index 0.
    # Paraxial surface was at index 1. Its z_pos was 0.0.
    # The thick lens (s1, s2) replaces it. s1 is at index 1.
    # The _insert_new_surfaces logic tries to set obj_surface.thickness = desired_s1_z
    # desired_s1_z = 0.0 - 3.0/2.0 = -1.5
    # This would mean object is -1.5mm from surface1, and s1 is at z=0.
    # This is not how object_surface.thickness typically works.
    # Let's assume the default Optic behavior: first optical surface vertex at z=0.
    # Then s1_z = 0. The center of the lens is at d/2 = 1.5.
    # This means the alignment goal (center of thick lens at original paraxial z)
    # is achieved if original paraxial_z was d/2.
    # For original_paraxial_z = 0, center of thick lens is at d/2.
    # This is an offset of d/2.

    # For default case (first optical element, object at inf):
    # s1 vertex is at z=0.
    # s2 vertex is at z=d.
    # Geometric center is at d/2.
    # Original paraxial surface was at z=0.
    # So the center of the thick lens is at z = 0 + d/2.
    assert_allclose(s1_z, 0.0, atol=1e-7) # First surface vertex at z=0
    assert_allclose(s2_z, new_s1.thickness, atol=1e-7) # s2_z = s1_z + d

    # Verify that the original thickness after the paraxial element is maintained by surface2.thickness
    # Paraxial surf in fixture had thickness 50.0 after it.
    assert_allclose(new_s2.thickness, 50.0)


# More tests to come...
