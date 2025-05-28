"""Unit tests for the SampledMTF class."""

import pytest
import numpy as np

import optiland.backend as be
from optiland.optic import Optic
from optiland.fields import Field
from optiland.wavelength import Wavelength
from optiland.materials import IdealMaterial
from optiland.surfaces import StandardSurface, ImageSurface, ObjectSurface
from optiland.geometries import Plane, Standard as StandardGeometry
from optiland.mtf import SampledMTF


def create_ideal_thin_lens_optic(focal_length=50.0, aperture_radius=10.0):
    """Creates a simple ideal thin lens Optic for testing.

    Args:
        focal_length (float, optional): The focal length of the thin lens in mm.
            Defaults to 50.0.
        aperture_radius (float, optional): The radius of the aperture in mm.
            Defaults to 10.0.

    Returns:
        Optic: An Optic instance representing the ideal thin lens system.
    """
    # Define materials
    air = IdealMaterial(name="air", refractive_index=1.0)
    glass = IdealMaterial(name="glass", refractive_index=1.5)

    # Surfaces
    object_surface = ObjectSurface(
        name="object",
        geometry=Plane(),
        material=air,
        is_stop=False,
        thickness=be.inf, # Object at infinity
    )

    # Thin lens formula: P = (n-1) * (1/R1 - 1/R2 + (n-1)*d/(n*R1*R2))
    # For a single surface thin lens in air, with R2 = infinity (plane): P = (n_lens - n_air) / R1
    # Power P = 1 / focal_length
    # So, R1 = (n_lens - n_air) * focal_length
    # Curvature c = 1 / R1
    lens_curvature = (glass.refractive_index[0] - air.refractive_index[0]) / focal_length

    lens_surface = StandardSurface(
        name="lens",
        geometry=StandardGeometry(curvature=lens_curvature, radius=aperture_radius), # Positive curvature for converging
        material=glass,
        is_stop=True, # Aperture stop
        thickness=5.0, # Arbitrary small thickness for "thin" lens representation, then image plane.
                       # This thickness is to the *next* surface in its material (glass)
                       # We'll place image surface in air, so need another surface or adjust.
                       # For simplicity, let's make the lens surface output to air.
    )
    # Re-defining lens_surface to output to air for simplicity of thin lens model
    # The "material" of a surface is what's *after* it.
    # So, the "lens_surface" is made of glass, and light propagates into air *after* it.
    # This means the power calculation needs the lens to be in air.
    # Let's adjust the model: Obj (air) -> S1 (glass, front of lens) -> S2 (air, back of lens) -> Image
    # For a true thin lens, we can approximate with one surface and set its material to air
    # and the thickness to the focal length.
    # The power is defined by the geometry and material *before* it.

    # Let's use a single surface with power. The "material" argument of StandardSurface
    # is the material *after* the surface.
    # The surface itself has its properties (curvature, aspheres).
    # Optic uses system-level material definitions.
    
    # Simpler approach: surface 0 (object), surface 1 (lens), surface 2 (image)
    # Material of object surface is 'air'. Thickness is infinity.
    # Material of lens surface is 'glass'. Thickness is small (e.g. 2mm).
    # Material of image surface is 'air'.

    # Let's define the lens surface again, assuming it's in air and its power comes from curvature
    # The material *after* the lens surface will be air, leading to the image.
    lens_surface_material_after = air # Light exits into air towards image plane

    lens_surface_redefined = StandardSurface(
        name="lens_front", # Represents the single effective surface of a thin lens
        geometry=StandardGeometry(curvature=lens_curvature, radius=aperture_radius),
        material=lens_surface_material_after, # Material light goes into *after* this surface
        is_stop=True,
        thickness=focal_length, # Place image surface at focal length in air
    )


    image_surface = ImageSurface(
        name="image",
        geometry=Plane(),
        # material=air, # Not needed for ImageSurface as it's the last one
    )

    # Wavelength and Field
    primary_wavelength = Wavelength(0.55) # 0.55 µm = 550 nm
    on_axis_field = Field(0, 0)

    # Create Optic
    # The first material in the list is for the object space.
    # The material of a surface s_i is materials[i].
    # The space between s_i and s_{i+1} has material materials[i].
    # The material *into which* light refracts at s_i is materials[i+1].
    
    # Let's re-think the simple thin lens for Optic structure:
    # Obj (in Air) -> Lens Surface (in Air, but has power) -> Image (in Air)
    # Material list: [air, air]
    # Surface list: [obj_surf, lens_surf, img_surf]
    # obj_surf material is air. thickness to lens_surf is large (e.g. 1e9 or inf handled by object_at_infinity)
    # lens_surf material is air. thickness to img_surf is focal_length.
    # This means the lens surface itself provides the power, without a change in refractive index *after* it for propagation.
    # This is how thin lenses are often modelled if power is directly assigned or curvature acts in current medium.
    # In Optic, refraction occurs from material[i] to material[i+1] at surface[i].
    # So, a thin lens with n_lens=1.5 in air (n_air=1.0) needs:
    # Air -> Surface1 (curvature c1) -> Glass -> Surface2 (curvature c2, thickness t_lens) -> Air -> Image
    # Power P = (n_g - n_a)*c1 + (n_a - n_g)*c2  (approx, ignoring t_lens for thin lens)
    # If R1 is front surface, R2 is back surface: P = (n_g-1)/R1 + (1-n_g)/R2 = (n_g-1)(1/R1 - 1/R2)
    # For a single surface providing power and then propagating to focus in same medium (air):
    # This is more like a Fresnel lens or a surface with magical properties.
    # Let's stick to a bi-convex lens with n=1.5, simplified to one surface for power.
    # Power P = 1/f. For one surface: P = (n_after - n_before) * c.
    # If lens is in air (n_before=1), made of glass (n_lens=1.5), then light enters glass.
    # So, n_after = 1.5. c_front = P_front / (1.5 - 1.0).
    # Then light exits glass. n_before_exit = 1.5, n_after_exit = 1.0.
    # c_back = P_back / (1.0 - 1.5).
    # For simplicity, let's assume the `StandardSurface` with `IdealMaterial` glass
    # implicitly handles the refraction from the *previous* material (air) into glass,
    # and its curvature acts on this. The `thickness` is then in glass.
    # Then we need another surface to exit glass into air.

    # Obj (air) --thickness_obj_to_lens1--> Lens1 (front, R1, into glass) --lens_thickness--> Lens2 (back, R2, into air) --dist_to_image--> Image
    
    # Let's use the provided example structure from other tests if available, or a very simple single powered surface.
    # The simplest is an object surface, one powered surface, and an image surface.
    # The powered surface will be in 'air' and its thickness will be 'focal_length' in 'air'.
    # This means the surface itself imparts the required phase change.

    object_s = ObjectSurface(name="object", material=air, geometry=Plane()) # Object at inf by default if first surface

    # For a thin lens in air, power P = 1/f.
    # If this power is achieved by a single surface, P = (n' - n) * C.
    # If we model the lens as existing in air (n=1, n'=1), then C must be infinite unless P=0.
    # This implies we must model the material change.
    # Surface 0: Object @ inf. Material before it is Air.
    # Surface 1: Lens front. Material before is Air. Material after is Glass. Curvature C1. Thickness T_lens.
    # Surface 2: Lens back. Material before is Glass. Material after is Air. Curvature C2. Thickness T_to_image.
    # Surface 3: Image.

    # Let's use equal convex-convex radii for simplicity. P_total = P1 + P2.
    # P1 = (n_g - n_a) / R; P2 = (n_a - n_g) / (-R) = (n_g - n_a) / R.
    # So P_total = 2 * (n_g - n_a) / R.
    # R = 2 * (n_g - n_a) / P_total = 2 * (n_g - n_a) * focal_length.
    # R_val = 2 * (glass.refractive_index[0] - air.refractive_index[0]) * focal_length
    # Curvature_val = 1 / R_val

    R_val = (glass.refractive_index[0] - air.refractive_index[0]) * focal_length # For a single surface lens
    # This is if P = (n_g - n_a) / R1.  So R1 = (n_g - n_a) * f.
    # This lens surface is assumed to transition from air to glass.
    
    lens_s1_curvature = 1.0 / R_val

    surf1 = StandardSurface(
        name="LensFront",
        geometry=StandardGeometry(curvature=lens_s1_curvature, radius=aperture_radius),
        material=glass, # Material *after* this surface is glass
        is_stop=True,
        thickness=2.0, # Small thickness for the lens itself
    )

    # Back surface of the lens. For simplicity, make it plane. Power comes from front.
    # This means focal_length needs to be calculated for a plano-convex lens.
    # P_plano_convex_R1 = (n_g - n_a) / R1.  So R1 = (n_g - n_a) * f. This is what we used.
    surf2 = StandardSurface(
        name="LensBack",
        geometry=Plane(), # Plano-convex
        material=air, # Material *after* this surface is air
        thickness=focal_length - surf1.thickness, # Approx, adjust to paraxial focus
                                                 # This thickness is in Air.
    )

    # Image surface
    image_s = ImageSurface(name="Image", geometry=Plane())

    # Optic setup
    optic = Optic(
        surfaces=[object_s, surf1, surf2, image_s],
        materials=[air, glass, air], # Object space, lens, image space
        fields=[on_axis_field],
        wavelengths=[primary_wavelength],
    )

    # Set image surface at paraxial focus
    paraxial_props = optic.paraxial
    # The last thickness in the surface list is ignored by paraxial calc, it uses image surface.
    # We need to set the thickness of the last material-filled space (surf2.thickness)
    # such that the image plane is at focus.
    # Paraxial calculation gives EFL, BFL. BFL is from last surface to focus.
    # So, surf2.thickness should be BFL.
    
    # With current setup:
    # ObjectSurface (implicit thickness from obj_at_inf)
    # surf1 (thickness in glass)
    # surf2 (thickness in air, this is what we set to BFL)
    # image_s
    
    # Let's try a simpler single-surface lens model again, as it's often used in tests.
    # Assume the StandardSurface itself acts as a thin lens of specified power,
    # and the material specified is for propagation *after* the lens.
    
    # Reset surfaces for a truly simple model
    c = 1.0 / (focal_length * (glass.refractive_index[0] - 1.0)) # c = (n-1)/f for lens in air, if R2 is flat. No, P = (n-1)/R1. So c = 1/R1 = P/(n-1) = 1/(f*(n-1))

    thin_lens_surface = StandardSurface(
        name="ThinLens",
        geometry=StandardGeometry(curvature=c, radius=aperture_radius),
        material=air, # Propagation space after lens is air
        is_stop=True,
        thickness=focal_length, # Image is at focal_length in air
    )
    
    # Optic setup
    optic_simple = Optic(
        surfaces=[
            ObjectSurface(name="Object", material=air, geometry=Plane()),
            thin_lens_surface,
            ImageSurface(name="Image", geometry=Plane())
        ],
        materials=[air, air], # Object space in air, image space in air. Lens is "thin"
        fields=[on_axis_field],
        wavelengths=[primary_wavelength],
    )
    # Paraxial calculation should place image surface correctly if thin_lens_surface.thickness is set by BFL.
    # The Optic class automatically sets the image surface to paraxial focus if last material is not set.
    # The thickness of the last surface with material is adjusted.
    # Here, thin_lens_surface.thickness is the one.
    
    # Let Optic place the image surface at paraxial focus.
    # The last specified thickness (thin_lens_surface.thickness) will be overridden by paraxial calc.
    # We need to ensure the material of thin_lens_surface is glass for refraction,
    # and then have an air space to the image.
    
    # Final simple model attempt:
    # Object (Air) -> Lens (Glass, front curve, small thickness) -> Image (Air, back plane of lens to image)
    
    lens_material = IdealMaterial("lens_glass", 1.5)
    air_material = IdealMaterial("env_air", 1.0)

    # R = (n-1)*f for plano-convex lens
    curvature_front = 1.0 / ((lens_material.refractive_index[0] - 1.0) * focal_length)

    s0 = ObjectSurface(material=air_material, geometry=Plane())
    s1 = StandardSurface(
        name="Lens",
        geometry=StandardGeometry(curvature=curvature_front, radius=aperture_radius),
        material=lens_material, # After this surface, light is in lens_material
        is_stop=True,
        thickness=1.0 # Minimal thickness for the lens body
    )
    s2 = StandardSurface(
        name="LensBackToAir", # Surface where light exits lens into air
        geometry=Plane(), # Plano-convex
        material=air_material, # After this surface, light is in air
        thickness=focal_length - s1.thickness # This needs to be BFL
    )
    s3 = ImageSurface(geometry=Plane())

    test_optic = Optic(
        surfaces=[s0, s1, s2, s3],
        materials=[air_material, lens_material, air_material],
        fields=[Field(0,0)],
        wavelengths=[Wavelength(0.550)] # um
    )
    
    # The Optic class constructor calls _calculate_paraxial_properties,
    # which sets image_surface_z based on BFL.
    # The thickness of the last material space (s2.thickness) is adjusted.
    # So, the value given (focal_length - s1.thickness) is more of a hint or starting point.
    # The actual focusing is handled by the paraxial solver.

    return test_optic


class TestSampledMTF:
    """Tests for the SampledMTF class."""

    def test_sampled_mtf_instantiation(self):
        """Test that SampledMTF can be instantiated without errors."""
        optic = create_ideal_thin_lens_optic()
        
        # On-axis field, primary wavelength from optic
        field = optic.fields.get_field_coords()[0]
        wavelength = optic.primary_wavelength # This is a Wavelength object, value is optic.primary_wavelength.value

        sampled_mtf_instance = SampledMTF(
            optic=optic,
            field=field, # Should be (0,0)
            wavelength=wavelength.value, # Pass the float value in um
            num_rays=32,
            distribution="hexapolar",
            zernike_terms=37,
            zernike_type="fringe"
        )
        assert sampled_mtf_instance is not None
        assert sampled_mtf_instance.optic == optic
        assert sampled_mtf_instance.field == field
        assert be.isclose(sampled_mtf_instance.wavelength, wavelength.value)
        assert sampled_mtf_instance.num_rays == 32

    # TODO: Add more tests for calculate_mtf, diffraction limited cases, etc.

    def test_mtf_at_zero_frequency(self):
        """Test that MTF at zero frequency is 1.0."""
        optic = create_ideal_thin_lens_optic()
        field = optic.fields.get_field_coords()[0]
        wavelength_obj = optic.primary_wavelength

        sampled_mtf_instance = SampledMTF(
            optic=optic,
            field=field,
            wavelength=wavelength_obj.value,
        )
        
        frequencies = [(0.0, 0.0)]
        mtf_values = sampled_mtf_instance.calculate_mtf(frequencies)
        
        assert len(mtf_values) == 1
        assert float(be.to_numpy(mtf_values[0])) == pytest.approx(1.0)

    def test_unaberrated_system_behavior(self):
        """Test MTF properties for a well-behaved (near diffraction-limited) system."""
        optic = create_ideal_thin_lens_optic(focal_length=50.0, aperture_radius=10.0) # FNO = 50/20 = 2.5
        field = optic.fields.get_field_coords()[0]
        wavelength_obj = optic.primary_wavelength # 0.55um

        # Nyquist for FNO=2.5, wl=0.55um is 1 / (0.00055mm * 2.5) = 727 cyc/mm
        # For FNO=5 (aperture_radius=5mm), Nyquist = 363 cyc/mm
        # Let's use aperture_radius=5 for FNO=5 to make frequencies more sensitive
        optic_fno5 = create_ideal_thin_lens_optic(focal_length=50.0, aperture_radius=5.0)


        sampled_mtf_instance = SampledMTF(
            optic=optic_fno5,
            field=field,
            wavelength=wavelength_obj.value,
            num_rays=64, # Increase for better sampling for unaberrated system
            zernike_terms=37 
        )
        
        # Frequencies should be well below Nyquist for an unaberrated system to show high MTF
        # but not too low that it's always ~1.0
        # Max freq for diffraction limited MTF (cutoff) = 1 / (wl_mm * FNO)
        # For FNO=5, wl=0.55um -> max_freq = 1 / (0.00055 * 5) = ~363 cycles/mm
        freqs = [(30.0, 0.0), (0.0, 30.0), (20.0, 20.0)] 
        mtf_values = sampled_mtf_instance.calculate_mtf(frequencies=freqs)

        assert len(mtf_values) == len(freqs)
        for mtf_val in mtf_values:
            val = float(be.to_numpy(mtf_val))
            assert 0.0 <= val <= 1.0
            # For a good system, MTF should be less than 1.0 for non-zero frequencies
            # and measurably high, but this depends heavily on sampling, num_rays etc.
            # For this test, primarily checking bounds and consistency.
            # With limited rays, it won't be perfectly diffraction limited.
            assert val < 1.0 

    def test_defocused_system_behavior(self):
        """Test that defocus lowers the MTF compared to a focused system."""
        focal_length = 50.0
        aperture_radius = 5.0 # FNO = 5
        optic_focused = create_ideal_thin_lens_optic(focal_length=focal_length, aperture_radius=aperture_radius)
        field = optic_focused.fields.get_field_coords()[0]
        wavelength_obj = optic_focused.primary_wavelength

        sampled_mtf_focused = SampledMTF(
            optic=optic_focused,
            field=field,
            wavelength=wavelength_obj.value,
            num_rays=32, # Keep num_rays modest for speed
        )
        
        freq = (25.0, 0.0) # A single non-zero frequency
        mtf_val_focused_list = sampled_mtf_focused.calculate_mtf(frequencies=[freq])
        mtf_val_focused = float(be.to_numpy(mtf_val_focused_list[0]))
        assert 0.0 <= mtf_val_focused <= 1.0

        # Create a defocused optic
        # The last surface with thickness before image is surf2 (LensBackToAir)
        # Its thickness is in air_material.
        # Original surfaces: [s0_obj, s1_lens_front, s2_lens_back, s3_image]
        # s2_lens_back.thickness is what Optic adjusts for focus.
        # To introduce defocus, we need to tell Optic the image is elsewhere,
        # or manually set the thickness and prevent Optic from re-adjusting.
        # The simplest way is to modify the thickness of the space before the ImageSurface
        # and re-initialize the optic. The paraxial calculation in Optic init will
        # use this modified thickness if we pass `image_surface_z=None` (default).
        # However, Optic calculates BFL and sets the thickness of the last material space.
        # So, we need to manually create the surfaces list with the desired defocus
        # and then ensure the image plane is where we want it relative to that.

        # Let's get the surfaces from the focused optic
        surfaces_orig = optic_focused.surfaces.get_surfaces()
        materials_orig = optic_focused.materials.get_materials() # list of Material objects
        
        # The surface before the image plane is surfaces_orig[-2] (s2: LensBackToAir)
        # Its thickness is what the paraxial solver sets to BFL.
        # We want to shift the image plane relative to this.
        # One way: change the thickness of this last airspace.
        
        defocus_amount = 0.5 # mm
        
        # Create new surface list with modified thickness
        # surf0, surf1 are same. surf2 thickness is changed. surf3 is image.
        s0_defocus = surfaces_orig[0]
        s1_defocus = surfaces_orig[1]
        s2_defocus_orig = surfaces_orig[2]

        # The thickness of s2_defocus_orig was set by paraxial solver to BFL.
        # We now want to evaluate at image_plane_z = BFL + defocus_amount.
        # So, s2_defocus.thickness should be BFL + defocus_amount.
        
        # Paraxial properties of focused optic
        bfl_focused = optic_focused.paraxial.BFL()

        s2_defocus = StandardSurface(
            name=s2_defocus_orig.name,
            geometry=s2_defocus_orig.geometry, # Plane
            material=s2_defocus_orig.material, # air_material
            is_stop=s2_defocus_orig.is_stop,
            thickness=bfl_focused + defocus_amount # New thickness
        )
        s3_defocus = surfaces_orig[3] # ImageSurface

        optic_defocused = Optic(
            surfaces=[s0_defocus, s1_defocus, s2_defocus, s3_defocus],
            materials=materials_orig, # Pass the list of Material objects
            fields=[field],
            wavelengths=[wavelength_obj],
            # By providing a full list of surfaces including ImageSurface, and the last
            # thickness being set, the paraxial solver won't override it if it matches
            # the image_surface_z it computes.
            # To be sure, we can force the image_surface_z.
            # However, Optic's _calculate_paraxial_properties will set image_surface_z
            # based on the new thickness. This is the desired behavior: the system
            # itself is defined with a defocused image plane.
        )

        sampled_mtf_defocused = SampledMTF(
            optic=optic_defocused,
            field=field,
            wavelength=wavelength_obj.value,
            num_rays=32,
        )
        
        mtf_val_defocused_list = sampled_mtf_defocused.calculate_mtf(frequencies=[freq])
        mtf_val_defocused = float(be.to_numpy(mtf_val_defocused_list[0]))

        assert 0.0 <= mtf_val_defocused < 1.0
        assert mtf_val_defocused < mtf_val_focused, \
            f"Defocused MTF ({mtf_val_defocused}) should be less than focused MTF ({mtf_val_focused})"


    def test_calculate_mtf_multiple_calls(self):
        """Test consistency of calculate_mtf with multiple calls."""
        optic = create_ideal_thin_lens_optic()
        field = optic.fields.get_field_coords()[0]
        wavelength_obj = optic.primary_wavelength

        sampled_mtf_instance = SampledMTF(
            optic=optic,
            field=field,
            wavelength=wavelength_obj.value,
        )

        freq1 = (5.0, 0.0)
        freq2 = (10.0, 0.0)

        mtf_val1_list = sampled_mtf_instance.calculate_mtf(frequencies=[freq1])
        mtf_val1 = float(be.to_numpy(mtf_val1_list[0]))

        mtf_val2_list = sampled_mtf_instance.calculate_mtf(frequencies=[freq2])
        mtf_val2 = float(be.to_numpy(mtf_val2_list[0]))
        
        mtf_vals_combined_list = sampled_mtf_instance.calculate_mtf(frequencies=[freq1, freq2])
        mtf_val1_combined = float(be.to_numpy(mtf_vals_combined_list[0]))
        mtf_val2_combined = float(be.to_numpy(mtf_vals_combined_list[1]))

        assert mtf_val1 == pytest.approx(mtf_val1_combined)
        assert mtf_val2 == pytest.approx(mtf_val2_combined)

    def test_zero_epd_handling(self, mocker):
        """Test MTF calculation when Exit Pupil Diameter (XPD) is zero."""
        optic = create_ideal_thin_lens_optic()
        field = optic.fields.get_field_coords()[0]
        wavelength_obj = optic.primary_wavelength

        # Mock the paraxial XPD call to return 0.0
        # The paraxial properties are on the Optic instance, specifically optic.paraxial object
        mocker.patch.object(optic.paraxial, 'XPD', return_value=0.0)
        # Ensure the mocked value is used by SampledMTF.
        # SampledMTF calls optic.paraxial.XPD() inside calculate_mtf.

        sampled_mtf_instance = SampledMTF(
            optic=optic, # This optic now has a mocked paraxial property
            field=field,
            wavelength=wavelength_obj.value,
        )
        
        frequencies = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (5.0, 5.0)]
        expected_mtfs = [1.0, 0.0, 0.0, 0.0]
        
        mtf_values = sampled_mtf_instance.calculate_mtf(frequencies=frequencies)
        
        assert len(mtf_values) == len(expected_mtfs)
        for i, mtf_val in enumerate(mtf_values):
            assert float(be.to_numpy(mtf_val)) == pytest.approx(expected_mtfs[i])
