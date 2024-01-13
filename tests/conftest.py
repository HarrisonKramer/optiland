import pytest
import numpy as np
from optiland import optic


@pytest.fixture()
def singlet_stop_surf1():
    lens = optic.Optic()
    lens.surface_group.surfaces = []

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=77.86103275, thickness=10.0,
                     material='N-SF11', is_stop=True)
    lens.add_surface(index=2, radius=4217.66563798, thickness=93.56549554)
    lens.add_surface(index=3)

    lens.set_aperture(aperture_type='EPD', value=25)

    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_field(y=3.5)
    lens.add_field(y=5)

    lens.add_wavelength(value=0.48613270)
    lens.add_wavelength(value=0.58756180, is_primary=True)
    lens.add_wavelength(value=0.65627250)

    lens.update_paraxial()

    return lens


@pytest.fixture()
def singlet_stop_surf2():
    lens = optic.Optic()
    lens.surface_group.surfaces = []

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=77.86103275, thickness=10.0,
                     material='N-SF11')
    lens.add_surface(index=2, radius=4217.66563798, thickness=93.56549554,
                     is_stop=True)
    lens.add_surface(index=3)

    lens.set_aperture(aperture_type='EPD', value=25)

    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_field(y=3.5)
    lens.add_field(y=5)

    lens.add_wavelength(value=0.48613270)
    lens.add_wavelength(value=0.58756180, is_primary=True)
    lens.add_wavelength(value=0.65627250)

    lens.update_paraxial()

    return lens


@pytest.fixture()
def cooke_triplet():
    pass


@pytest.fixture()
def telephoto():
    pass


@pytest.fixture()
def double_gauss():
    pass


@pytest.fixture()
def hubble():
    pass
