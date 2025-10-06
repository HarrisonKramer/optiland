# tests/optic/test_optic.py
"""
Tests for the main Optic class in optiland.optic.
"""
import pytest
from unittest.mock import patch
import optiland.backend as be

from optiland.apodization import GaussianApodization
from optiland.aperture import Aperture
from optiland.fields import FieldGroup, AngleField
from optiland.optic import Optic
from optiland.rays import create_polarization
from optiland.samples.objectives import HeliarLens
from optiland.surfaces import SurfaceGroup
from optiland.wavelength import WavelengthGroup
from ..utils import assert_allclose
from optiland.surfaces.factories.material_factory import MaterialFactory


def singlet_infinite_object():
    """
    Creates a simple singlet lens optic with an object at infinity.
    Used for testing optic combination.
    """
    lens = Optic()
    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(
        index=1,
        thickness=7,
        radius=43.7354,
        is_stop=True,
        material="N-SF11",
    )
    lens.add_surface(index=2, radius=-46.2795, thickness=50)
    lens.add_surface(index=3)

    lens.set_aperture(aperture_type="EPD", value=25)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_wavelength(value=0.5, is_primary=True)
    return lens


def singlet_finite_object():
    """
    Creates a simple singlet lens optic with a finite object distance.
    Used for testing optic combination.
    """
    lens = Optic()
    lens.add_surface(index=0, radius=be.inf, thickness=50)
    lens.add_surface(
        index=1,
        thickness=7,
        radius=43.7354,
        is_stop=True,
        material="N-SF11",
    )
    lens.add_surface(index=2, radius=-46.2795, thickness=50)
    lens.add_surface(index=3)

    lens.set_aperture(aperture_type="EPD", value=25)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_wavelength(value=0.5, is_primary=True)
    return lens


class TestOptic:
    """
    Tests the main Optic class, which serves as the primary interface for
    creating and manipulating optical systems.
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initializes a fresh Optic instance for each test."""
        self.optic = Optic()

    def test_initialization(self, set_test_backend):
        """
        Tests that an Optic instance is initialized with the correct default
        attributes.
        """
        assert self.optic.aperture is None
        assert self.optic.field_definition is None
        assert isinstance(self.optic.surface_group, SurfaceGroup)
        assert isinstance(self.optic.fields, FieldGroup)
        assert isinstance(self.optic.wavelengths, WavelengthGroup)
        assert self.optic.polarization == "ignore"
        assert not self.optic.obj_space_telecentric

    def test_add_surface(self, set_test_backend):
        """Tests that a surface can be added to the optic."""
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        assert len(self.optic.surface_group.surfaces) == 1

    def test_add_field(self, set_test_backend):
        """Tests that a field point can be added to the optic."""
        self.optic.add_field(10.0, 5.0)
        assert len(self.optic.fields.fields) == 1
        assert self.optic.fields.fields[0].y == 10.0
        assert self.optic.fields.fields[0].x == 5.0

    def test_add_wavelength(self, set_test_backend):
        """Tests that a wavelength can be added to the optic."""
        self.optic.add_wavelength(0.55, is_primary=True)
        assert len(self.optic.wavelengths.wavelengths) == 1
        assert self.optic.wavelengths.wavelengths[0].value == 0.55
        assert self.optic.wavelengths.wavelengths[0].is_primary

    def test_set_aperture(self, set_test_backend):
        """Tests that the system aperture can be set correctly."""
        self.optic.set_aperture("EPD", 5.0)
        assert isinstance(self.optic.aperture, Aperture)
        assert self.optic.aperture.ap_type == "EPD"
        assert self.optic.aperture.value == 5.0

    def test_set_field_type(self, set_test_backend):
        """Tests that the field type (e.g., angle, height) can be set."""
        self.optic.set_field_type("angle")
        assert isinstance(self.optic.field_definition, AngleField)

    def test_set_comment(self, set_test_backend):
        """Tests that comments can be added to surfaces."""
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
            comment="Object surface",
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="SF11",
            thickness=5,
            comment="First surface",
        )

        assert self.optic.surface_group.surfaces[0].comment == "Object surface"
        assert self.optic.surface_group.surfaces[1].comment == "First surface"

    def test_set_radius(self, set_test_backend):
        """Tests that the radius of a surface can be modified."""
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.set_radius(10.0, 0)
        assert self.optic.surface_group.surfaces[0].geometry.radius == 10.0

    def test_set_conic(self, set_test_backend):
        """Tests that the conic constant of a surface can be modified."""
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.set_conic(-1.0, 0)
        assert self.optic.surface_group.surfaces[0].geometry.k == -1.0

    def test_set_thickness(self, set_test_backend):
        """Tests that the thickness of a surface can be modified."""
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.add_surface(
            index=2,
            surface_type="standard",
            material="air",
            thickness=10,
        )
        self.optic.set_thickness(10.0, 1)
        assert self.optic.surface_group.get_thickness(1) == 10.0

    def test_set_index(self, set_test_backend):
        """
        Tests that the refractive index of a surface's material can be
        modified directly.
        """
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=2,
            surface_type="standard",
            material="air",
            thickness=10,
        )
        self.optic.set_index(1.5, 1)
        assert self.optic.surface_group.surfaces[1].material_post.n(1) == 1.5

    def test_set_material(self, set_test_backend):
        """Tests that the material of a surface can be replaced."""
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=2,
            surface_type="standard",
            material="air",
            thickness=10,
        )
        surface_number = 1
        material_post = MaterialFactory._configure_post_material("N-BK7")
        self.optic.set_material(material_post, surface_number)
        surface = self.optic.surface_group.surfaces[surface_number]
        assert surface.material_post == material_post

    def test_set_asphere_coeff(self, set_test_backend):
        """Tests that aspheric coefficients of a surface can be modified."""
        self.optic.add_surface(
            index=0,
            surface_type="even_asphere",
            material="air",
            thickness=5,
            coefficients=[0.0, 0.0, 0.0],
        )
        self.optic.set_asphere_coeff(0.1, 0, 2)
        assert self.optic.surface_group.surfaces[0].geometry.coefficients[2] == 0.1

    def test_set_polarization(self, set_test_backend):
        """Tests that the polarization handling state can be set."""
        self.optic.set_polarization("ignore")
        assert self.optic.polarization == "ignore"

    def test_optic_default_apodization(self, set_test_backend):
        """Tests that the default apodization is None."""
        assert self.optic.apodization is None

    def test_optic_set_apodization(self, set_test_backend):
        """Tests that apodization can be set on the optic."""
        gaussian_apod = GaussianApodization(sigma=0.5)
        self.optic.set_apodization(gaussian_apod)
        assert self.optic.apodization == gaussian_apod, "Apodization not set correctly"

        # Test setting with a non-Apodization type
        with pytest.raises(TypeError):
            self.optic.set_apodization("not_an_apodization_object")

    def test_set_invalid_polarization(self, set_test_backend):
        """
        Tests that setting an invalid polarization state raises a ValueError.
        """
        with pytest.raises(ValueError):
            self.optic.set_polarization("invalid")

    def test_set_pickup(self, set_test_backend):
        """Tests that a pickup constraint can be added to the optic."""
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.pickups.add(0, "radius", 1, scale=2, offset=1)
        assert len(self.optic.pickups) == 1

    def test_clear_pickups(self, set_test_backend):
        """Tests that all pickup constraints can be cleared."""
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.pickups.add(0, "radius", 1, scale=2, offset=1)
        self.optic.pickups.clear()
        assert len(self.optic.pickups) == 0

    def test_set_solve(self, set_test_backend):
        """Tests that a solve can be added to the optic."""
        optic = HeliarLens()
        optic.solves.add("marginal_ray_height", 6, height=10)
        assert len(optic.solves) == 1

    def test_clear_solves(self, set_test_backend):
        """Tests that all solves can be cleared."""
        optic = HeliarLens()
        optic.solves.add("marginal_ray_height", 6, height=10)
        optic.solves.clear()
        assert len(optic.solves) == 0

    def test_scale_system(self, set_test_backend):
        """
        Tests that the entire optical system can be scaled by a factor.
        """
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        self.optic.set_aperture("EPD", 5.0)
        self.optic.scale_system(2)
        assert self.optic.surface_group.surfaces[0].geometry.radius == 2 * 10.0
        assert self.optic.aperture.value == 2 * 5.0

    def test_reset(self, set_test_backend):
        """
        Tests that the optic can be reset to its initial, empty state.
        """
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        self.optic.reset()
        assert self.optic.aperture is None
        assert self.optic.field_definition is None
        assert len(self.optic.surface_group.surfaces) == 0
        assert len(self.optic.fields.fields) == 0
        assert len(self.optic.wavelengths.wavelengths) == 0
        assert len(self.optic.pickups) == 0
        assert len(self.optic.solves) == 0

    def test_n(self, set_test_backend):
        """
        Tests the `n` method, which retrieves the refractive indices for all
        materials at the primary wavelength.
        """
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        self.optic.add_wavelength(0.55, is_primary=True)
        n_values = self.optic.n()
        assert len(n_values) == 1

    def test_update_paraxial(self, set_test_backend):
        """
        Tests that the paraxial properties of the system can be updated without
        error.
        """
        lens = HeliarLens()
        lens.update_paraxial()
        # This test just ensures the method runs without error.

    def test_update(self, set_test_backend):
        """
        Tests that the system can be fully updated (solves, pickups, etc.)
        without error.
        """
        lens = HeliarLens()
        lens.update()
        # This test just ensures the method runs without error.

    def test_image_solve(self, set_test_backend):
        """
        Tests that the automatic image distance solve can be performed
        without error.
        """
        lens = HeliarLens()
        lens.image_solve()
        # This test just ensures the method runs without error.

    def test_trace(self, set_test_backend):
        """
        Tests standard ray tracing through a known system (HeliarLens).
        """
        lens = HeliarLens()
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_generic(self, set_test_backend):
        """
        Tests generic ray tracing with normalized coordinates through a
        known system.
        """
        lens = HeliarLens()
        rays = lens.trace_generic(0.0, 0.0, 0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_invalid_field(self, set_test_backend):
        """
        Tests that ray tracing with an invalid field coordinate raises a
        ValueError.
        """
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace(0.0, 2.0, 0.55)

    def test_trace_generic_invalid_field(self, set_test_backend):
        """
        Tests that generic ray tracing with an invalid field coordinate
        raises a ValueError.
        """
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace_generic(0.0, 0.0, 0.0, 2.0, 0.55)

    def test_trace_generic_invalid_pupil(self, set_test_backend):
        """
        Tests that generic ray tracing with an invalid pupil coordinate
        raises a ValueError.
        """
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace_generic(0.0, 5.0, 0.0, 0.0, 0.55)

    def test_trace_polarized(self, set_test_backend):
        """
        Tests that polarized ray tracing can be performed without error.
        """
        lens = HeliarLens()
        state = create_polarization("unpolarized")
        lens.set_polarization(state)
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_object_property(self, set_test_backend):
        """Tests that the object_surface property is correctly retrieved."""
        assert self.optic.object_surface is None
        self.optic.add_surface(index=0)
        assert self.optic.object_surface is self.optic.surface_group.surfaces[0]


    def test_image_surface_property(self, set_test_backend):
        """Tests that the image_surface property is correctly retrieved."""
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        assert self.optic.image_surface is self.optic.surface_group.surfaces[0]

    def test_total_track_property(self, set_test_backend):
        """Tests that the total track length of the system is correct."""
        lens = HeliarLens()
        assert lens.total_track == 12.1357

    def test_total_track_error(self, set_test_backend):
        """
        Tests that accessing total_track raises a ValueError for an invalid
        system (e.g., only one surface).
        """
        lens = HeliarLens()
        # manually remove all but first surface
        lens.surface_group.surfaces = [lens.surface_group.surfaces[0]]
        with pytest.raises(ValueError):
            _ = lens.total_track

    def test_polarization_state_property(self, set_test_backend):
        """
        Tests that the polarization_state property correctly reflects the
        set polarization.
        """
        lens = HeliarLens()
        assert lens.polarization_state is None

        state = create_polarization("unpolarized")
        lens.set_polarization(state)
        assert lens.polarization_state == state

    def test_polarization_state_error(self, set_test_backend):
        """
        Tests that accessing polarization_state with an invalid internal
        state raises a ValueError.
        """
        lens = HeliarLens()
        lens.polarization = "invalid"
        with pytest.raises(ValueError):
            _ = lens.polarization_state

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of an Optic instance to a dictionary.
        """
        lens = HeliarLens()
        lens.set_apodization(GaussianApodization(sigma=0.5))
        lens_dict = lens.to_dict()
        assert isinstance(lens_dict, dict)
        assert "apodization" in lens_dict
        assert lens_dict["apodization"]["type"] == "GaussianApodization"
        assert lens_dict["apodization"]["sigma"] == 0.5

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of an Optic instance from a dictionary.
        """
        lens = HeliarLens()
        lens.set_apodization(GaussianApodization(sigma=0.5))
        basic_dict = lens.to_dict()

        new_optic = Optic.from_dict(basic_dict)
        assert isinstance(new_optic, Optic)
        assert isinstance(new_optic.apodization, GaussianApodization)

        # Test deserialization without apodization data
        basic_dict.pop("apodization", None)
        new_optic = Optic.from_dict(basic_dict)
        assert new_optic.apodization is None

    def test_invalid_field_type(self, set_test_backend):
        """
        Tests that setting an invalid field type raises a ValueError.
        """
        with pytest.raises(ValueError):
            self.optic.set_field_type("invalid")

    def test_no_stop(self, set_test_backend):
        """
        Tests that accessing the stop surface index when no stop is defined
        raises a ValueError.
        """
        for surface in self.optic.surface_group.surfaces:
            surface.is_stop = False
        with pytest.raises(ValueError):
            _ = self.optic.surface_group.stop_index

    def test_add_infinite_object(self):  # do not test for torch backend
        """
        Tests the combination of two optical systems, where the object is at
        infinity.
        """
        lens1 = singlet_infinite_object()
        lens2 = singlet_infinite_object()
        lens_combined = lens1 + lens2
        assert lens_combined.surface_group.num_surfaces == 6

        # test that a ray trace through the combined lens works
        rays = lens_combined.trace(
            Hx=0, Hy=0, distribution="random", num_rays=42, wavelength=0.5
        )
        assert rays is not None

    def test_add_finite_object(self):  # do not test for torch backend
        """
        Tests the combination of two optical systems with finite object
        distances.
        """
        lens1 = singlet_finite_object()
        lens2 = singlet_finite_object()
        lens_combined = lens1 + lens2
        assert lens_combined.surface_group.num_surfaces == 6

        # test that a ray trace through the combined lens works
        rays = lens_combined.trace(
            Hx=0, Hy=0, distribution="random", num_rays=42, wavelength=0.5
        )
        assert rays is not None

    def test_invalid_coordinate_system(self, set_test_backend):
        """
        Tests that defining surfaces with an inconsistent coordinate system
        (e.g., mixing absolute and relative positions) raises a ValueError.
        """
        with pytest.raises(ValueError):
            self.optic.add_surface(index=0, radius=be.inf, z=-100)
            self.optic.add_surface(
                index=1, radius=be.inf, z=0, dx=15
            )  # cannot use dx or dy with abs. z

    def test_flip_optic(self, set_test_backend):
        """
        Tests that an entire optical system can be flipped (reversed).
        """
        lens = HeliarLens()
        lens.surface_group.set_fresnel_coatings()
        radii_orig = be.copy(lens.surface_group.radii)
        radii_orig = radii_orig[~be.isinf(radii_orig)]  # ignore inf
        n_orig = be.copy(lens.n(0.55))
        lens.flip()
        radii_flipped = be.copy(lens.surface_group.radii)
        radii_flipped = radii_flipped[~be.isinf(radii_flipped)]  # ignore inf
        n_flipped = be.copy(lens.n(0.55))
        assert_allclose(radii_orig, -be.flip(radii_flipped))
        assert_allclose(n_orig[:-1], be.flip(n_flipped[:-1]))

    def test_invalid_flip(self, set_test_backend):
        """
        Tests that flipping an empty optic raises a ValueError.
        """
        lens = Optic()
        with pytest.raises(ValueError):
            lens.flip()

    def test_flip_solves_pickups(self, set_test_backend):
        """
        Tests that solves and pickups are correctly updated when an optic is
        flipped.
        """
        lens = Optic()

        lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
        lens.add_surface(
            index=1, radius=100, thickness=4, material="SK16", is_stop=True
        )
        lens.add_surface(index=2, radius=-1000, thickness=20)
        lens.add_surface(index=3)
        lens.set_aperture(aperture_type="EPD", value=10.0)
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        lens.add_wavelength(value=0.5876, is_primary=True)

        lens.solves.add("quick_focus")
        lens.pickups.add(
            source_surface_idx=1,
            attr_type="radius",
            target_surface_idx=2,
            scale=-1,
            offset=0,
        )
        lens.update()
        lens.flip()
        assert lens.pickups.pickups[0].source_surface_idx == 2
        assert lens.pickups.pickups[0].target_surface_idx == 1

    @patch("optiland.optic.optic.SurfaceSagViewer")
    def test_plot_surface_sag(self, mock_viewer, set_test_backend):
        """
        Tests the plotting of a surface sag by ensuring the correct viewer
        class is called with the correct parameters.
        """
        lens = singlet_infinite_object()

        lens.plot_surface_sag(
            surface_index=1, y_cross_section=2.0, x_cross_section=-2.0
        )
        mock_viewer.assert_called_once_with(lens)
        viewer_instance = mock_viewer.return_value
        viewer_instance.view.assert_called_once_with(1, 2.0, -2.0)