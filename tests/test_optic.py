import pytest
from optiland.optic import Optic
from optiland.fields import FieldGroup
from optiland.surfaces import SurfaceGroup
from optiland.wavelength import WavelengthGroup
from optiland.aperture import Aperture
from optiland.samples.objectives import HeliarLens
from optiland.rays import create_polarization


class TestOptic:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Optic()

    def test_initialization(self):
        assert self.optic.aperture is None
        assert self.optic.field_type is None
        assert isinstance(self.optic.surface_group, SurfaceGroup)
        assert isinstance(self.optic.fields, FieldGroup)
        assert isinstance(self.optic.wavelengths, WavelengthGroup)
        assert self.optic.polarization == 'ignore'
        assert not self.optic.obj_space_telecentric

    def test_add_surface(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='SF11', thickness=5)
        assert len(self.optic.surface_group.surfaces) == 1

    def test_add_field(self):
        self.optic.add_field(10.0, 5.0)
        assert len(self.optic.fields.fields) == 1
        assert self.optic.fields.fields[0].y == 10.0
        assert self.optic.fields.fields[0].x == 5.0

    def test_add_wavelength(self):
        self.optic.add_wavelength(0.55, is_primary=True)
        assert len(self.optic.wavelengths.wavelengths) == 1
        assert self.optic.wavelengths.wavelengths[0].value == 0.55
        assert self.optic.wavelengths.wavelengths[0].is_primary

    def test_set_aperture(self):
        self.optic.set_aperture('EPD', 5.0)
        assert isinstance(self.optic.aperture, Aperture)
        assert self.optic.aperture.ap_type == 'EPD'
        assert self.optic.aperture.value == 5.0

    def test_set_field_type(self):
        self.optic.set_field_type('angle')
        assert self.optic.field_type == 'angle'

    def test_set_radius(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='SF11', thickness=5)
        self.optic.set_radius(10.0, 0)
        assert self.optic.surface_group.surfaces[0].geometry.radius == 10.0

    def test_set_conic(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='SF11', thickness=5)
        self.optic.set_conic(-1.0, 0)
        assert self.optic.surface_group.surfaces[0].geometry.k == -1.0

    def test_set_thickness(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='air', thickness=5)
        self.optic.add_surface(index=1, surface_type='standard',
                               material='SF11', thickness=5)
        self.optic.add_surface(index=2, surface_type='standard',
                               material='air', thickness=10)
        self.optic.set_thickness(10.0, 1)
        assert self.optic.surface_group.get_thickness(1) == 10.0

    def test_set_index(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='air', thickness=5)
        self.optic.add_surface(index=1, surface_type='standard',
                               material='air', thickness=5)
        self.optic.add_surface(index=2, surface_type='standard',
                               material='air', thickness=10)
        self.optic.set_index(1.5, 1)
        assert self.optic.surface_group.surfaces[1].material_post.n(1) == 1.5

    def test_set_asphere_coeff(self):
        self.optic.add_surface(index=0, surface_type='even_asphere',
                               material='air', thickness=5,
                               coefficients=[0.0, 0.0, 0.0])
        self.optic.set_asphere_coeff(0.1, 0, 2)
        assert self.optic.surface_group.surfaces[0].geometry.c[2] == 0.1

    def test_set_polarization(self):
        self.optic.set_polarization('ignore')
        assert self.optic.polarization == 'ignore'

    def test_set_invalid_polarization(self):
        with pytest.raises(ValueError):
            self.optic.set_polarization('invalid')

    def test_set_pickup(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='air', thickness=5)
        self.optic.add_surface(index=1, surface_type='standard',
                               material='air', thickness=5)
        self.optic.pickups.add(0, 'radius', 1, scale=2, offset=1)
        assert len(self.optic.pickups) == 1

    def test_clear_pickups(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='air', thickness=5)
        self.optic.add_surface(index=1, surface_type='standard',
                               material='air', thickness=5)
        self.optic.pickups.add(0, 'radius', 1, scale=2, offset=1)
        self.optic.pickups.clear()
        assert len(self.optic.pickups) == 0

    def test_set_solve(self):
        optic = HeliarLens()
        optic.solves.add('marginal_ray_height', 6, height=10)
        assert len(optic.solves) == 1

    def test_clear_solves(self):
        optic = HeliarLens()
        optic.solves.add('marginal_ray_height', 6, height=10)
        optic.solves.clear()
        assert len(optic.solves) == 0

    def test_scale_system(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='air', radius=10, thickness=5)
        self.optic.set_aperture('EPD', 5.0)
        self.optic.scale_system(2)
        assert self.optic.surface_group.surfaces[0].geometry.radius == 2 * 10.0
        assert self.optic.aperture.value == 2 * 5.0

    def test_reset(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='air', radius=10, thickness=5)
        self.optic.reset()
        assert self.optic.aperture is None
        assert self.optic.field_type is None
        assert len(self.optic.surface_group.surfaces) == 0
        assert len(self.optic.fields.fields) == 0
        assert len(self.optic.wavelengths.wavelengths) == 0
        assert len(self.optic.pickups) == 0
        assert len(self.optic.solves) == 0

    def test_n(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='air', radius=10, thickness=5)
        self.optic.add_wavelength(0.55, is_primary=True)
        n_values = self.optic.n()
        assert len(n_values) == 1

    def test_update_paraxial(self):
        lens = HeliarLens()
        lens.update_paraxial()
        # This test is just to ensure the method runs without error

    def test_update(self):
        lens = HeliarLens()
        lens.update()
        # This test is just to ensure the method runs without error

    def test_image_solve(self):
        lens = HeliarLens()
        lens.image_solve()
        # This test is just to ensure the method runs without error

    def test_trace(self):
        lens = HeliarLens()
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_generic(self):
        lens = HeliarLens()
        rays = lens.trace_generic(0.0, 0.0, 0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_invalid_field(self):
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace(0.0, 2.0, 0.55)

    def test_trace_generic_invalid_field(self):
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace_generic(0.0, 0.0, 0.0, 2.0, 0.55)

    def test_trace_generic_invalid_pupil(self):
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace_generic(0.0, 5.0, 0.0, 0.0, 0.55)

    def test_trace_polarized(self):
        lens = HeliarLens()
        state = create_polarization('unpolarized')
        lens.set_polarization(state)
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_object_property(self):
        assert self.optic.object_surface is None

    def test_image_surface_property(self):
        self.optic.add_surface(index=0, surface_type='standard',
                               material='air', radius=10, thickness=5)
        assert self.optic.image_surface is self.optic.surface_group.surfaces[0]

    def test_total_track_property(self):
        lens = HeliarLens()
        assert lens.total_track == 3.6291

    def test_polarization_state_property(self):
        lens = HeliarLens()
        assert lens.polarization_state is None

        state = create_polarization('unpolarized')
        lens.set_polarization(state)
        assert lens.polarization_state == state

    def test_to_dict(self):
        lens = HeliarLens()
        lens_dict = lens.to_dict()
        assert lens_dict is not None

    def test_from_dict(self):
        lens = HeliarLens()
        lens_dict = lens.to_dict()
        new_lens = HeliarLens.from_dict(lens_dict)
        assert new_lens is not None
        assert new_lens.total_track == lens.total_track

    def test_invalid_field_type(self):
        with pytest.raises(ValueError):
            self.optic.set_field_type('invalid')

    def test_no_stop(self):
        for surface in self.optic.surface_group.surfaces:
            surface.is_stop = False
        with pytest.raises(ValueError):
            self.optic.surface_group.stop_index
