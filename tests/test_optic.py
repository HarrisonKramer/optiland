import pytest
import optiland.backend as be

from optiland.aperture import Aperture
from optiland.fields import FieldGroup
from optiland.optic import Optic
from optiland.rays import create_polarization
from optiland.samples.objectives import HeliarLens
from optiland.surfaces import SurfaceGroup
from optiland.wavelength import WavelengthGroup
from tests.utils import assert_allclose


def singlet_infinite_object():
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
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Optic()

    def test_initialization(self, set_test_backend):
        assert self.optic.aperture is None
        assert self.optic.field_type is None
        assert isinstance(self.optic.surface_group, SurfaceGroup)
        assert isinstance(self.optic.fields, FieldGroup)
        assert isinstance(self.optic.wavelengths, WavelengthGroup)
        assert self.optic.polarization == "ignore"
        assert not self.optic.obj_space_telecentric

    def test_add_surface(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        assert len(self.optic.surface_group.surfaces) == 1

    def test_add_field(self, set_test_backend):
        self.optic.add_field(10.0, 5.0)
        assert len(self.optic.fields.fields) == 1
        assert self.optic.fields.fields[0].y == 10.0
        assert self.optic.fields.fields[0].x == 5.0

    def test_add_wavelength(self, set_test_backend):
        self.optic.add_wavelength(0.55, is_primary=True)
        assert len(self.optic.wavelengths.wavelengths) == 1
        assert self.optic.wavelengths.wavelengths[0].value == 0.55
        assert self.optic.wavelengths.wavelengths[0].is_primary

    def test_set_aperture(self, set_test_backend):
        self.optic.set_aperture("EPD", 5.0)
        assert isinstance(self.optic.aperture, Aperture)
        assert self.optic.aperture.ap_type == "EPD"
        assert self.optic.aperture.value == 5.0

    def test_set_field_type(self, set_test_backend):
        self.optic.set_field_type("angle")
        assert self.optic.field_type == "angle"

    def test_set_comment(self, set_test_backend):
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
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.set_radius(10.0, 0)
        assert self.optic.surface_group.surfaces[0].geometry.radius == 10.0

    def test_set_conic(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.set_conic(-1.0, 0)
        assert self.optic.surface_group.surfaces[0].geometry.k == -1.0

    def test_set_thickness(self, set_test_backend):
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

    def test_set_asphere_coeff(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="even_asphere",
            material="air",
            thickness=5,
            coefficients=[0.0, 0.0, 0.0],
        )
        self.optic.set_asphere_coeff(0.1, 0, 2)
        assert self.optic.surface_group.surfaces[0].geometry.c[2] == 0.1

    def test_set_polarization(self, set_test_backend):
        self.optic.set_polarization("ignore")
        assert self.optic.polarization == "ignore"

    def test_set_invalid_polarization(self, set_test_backend):
        with pytest.raises(ValueError):
            self.optic.set_polarization("invalid")

    def test_set_pickup(self, set_test_backend):
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
        optic = HeliarLens()
        optic.solves.add("marginal_ray_height", 6, height=10)
        assert len(optic.solves) == 1

    def test_clear_solves(self, set_test_backend):
        optic = HeliarLens()
        optic.solves.add("marginal_ray_height", 6, height=10)
        optic.solves.clear()
        assert len(optic.solves) == 0

    def test_scale_system(self, set_test_backend):
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
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        self.optic.reset()
        assert self.optic.aperture is None
        assert self.optic.field_type is None
        assert len(self.optic.surface_group.surfaces) == 0
        assert len(self.optic.fields.fields) == 0
        assert len(self.optic.wavelengths.wavelengths) == 0
        assert len(self.optic.pickups) == 0
        assert len(self.optic.solves) == 0

    def test_n(self, set_test_backend):
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
        lens = HeliarLens()
        lens.update_paraxial()
        # This test is just to ensure the method runs without error

    def test_update(self, set_test_backend):
        lens = HeliarLens()
        lens.update()
        # This test is just to ensure the method runs without error

    def test_image_solve(self, set_test_backend):
        lens = HeliarLens()
        lens.image_solve()
        # This test is just to ensure the method runs without error

    def test_trace(self, set_test_backend):
        lens = HeliarLens()
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_generic(self, set_test_backend):
        lens = HeliarLens()
        rays = lens.trace_generic(0.0, 0.0, 0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_invalid_field(self, set_test_backend):
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace(0.0, 2.0, 0.55)

    def test_trace_generic_invalid_field(self, set_test_backend):
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace_generic(0.0, 0.0, 0.0, 2.0, 0.55)

    def test_trace_generic_invalid_pupil(self, set_test_backend):
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace_generic(0.0, 5.0, 0.0, 0.0, 0.55)

    def test_trace_polarized(self, set_test_backend):
        lens = HeliarLens()
        state = create_polarization("unpolarized")
        lens.set_polarization(state)
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_object_property(self, set_test_backend):
        assert self.optic.object_surface is None

    def test_image_surface_property(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        assert self.optic.image_surface is self.optic.surface_group.surfaces[0]

    def test_total_track_property(self, set_test_backend):
        lens = HeliarLens()
        assert lens.total_track == 3.6291

    def test_total_track_error(self, set_test_backend):
        lens = HeliarLens()
        # manually remove all but first surface
        lens.surface_group.surfaces = [lens.surface_group.surfaces[0]]
        with pytest.raises(ValueError):
            _ = lens.total_track

    def test_polarization_state_property(self, set_test_backend):
        lens = HeliarLens()
        assert lens.polarization_state is None

        state = create_polarization("unpolarized")
        lens.set_polarization(state)
        assert lens.polarization_state == state

    def test_polarization_state_error(self, set_test_backend):
        lens = HeliarLens()
        lens.polarization = "invalid"
        with pytest.raises(ValueError):
            _ = lens.polarization_state

    def test_to_dict(self, set_test_backend):
        lens = HeliarLens()
        lens_dict = lens.to_dict()
        assert lens_dict is not None

    def test_from_dict(self, set_test_backend):
        lens = HeliarLens()
        lens_dict = lens.to_dict()
        new_lens = HeliarLens.from_dict(lens_dict)
        assert new_lens is not None
        assert new_lens.total_track == lens.total_track

    def test_invalid_field_type(self, set_test_backend):
        with pytest.raises(ValueError):
            self.optic.set_field_type("invalid")

    def test_no_stop(self, set_test_backend):
        for surface in self.optic.surface_group.surfaces:
            surface.is_stop = False
        with pytest.raises(ValueError):
            _ = self.optic.surface_group.stop_index

    def test_add_infinite_object(self):  # do not test for torch backend
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
        with pytest.raises(ValueError):
            self.optic.add_surface(index=0, radius=be.inf, z=-100)
            self.optic.add_surface(
                index=1, radius=be.inf, z=0, dx=15
            )  # cannot use dx or dy with abs. z

    def test_flip_optic(self, set_test_backend):
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
        lens = Optic()
        with pytest.raises(ValueError):
            lens.flip()

    def test_flip_solves_pickups(self, set_test_backend):
        lens = Optic()

        lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
        lens.add_surface(index=1, radius=100, thickness=4, material="SK16", is_stop=True)
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
