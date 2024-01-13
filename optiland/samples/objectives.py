import numpy as np
from optiland import optic


def TripletTelescopeObjective():
    lens = optic.Optic()

    lens.surface_group.surfaces = []

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=50.098, thickness=4.5,
                     material='N-BK7', is_stop=True)
    lens.add_surface(index=2, radius=-983.42, thickness=0.1)
    lens.add_surface(index=3, radius=56.671, thickness=4.5,
                     material='N-BK7')
    lens.add_surface(index=4, radius=-171.15, thickness=5.571)
    lens.add_surface(index=5, radius=-97.339, thickness=3.5,
                     material=('SF1', 'schott'))
    lens.add_surface(index=6, radius=81.454, thickness=75.132)
    lens.add_surface(index=7)

    lens.set_aperture(aperture_type='imageFNO', value=2.8)

    lens.set_field_type(field_type='angle')
    lens.add_field(y=0.0)
    lens.add_field(y=0.7)
    lens.add_field(y=1.0)

    lens.add_wavelength(value=0.4861)
    lens.add_wavelength(value=0.5876, is_primary=True)
    lens.add_wavelength(value=0.6563)

    lens.update_paraxial()

    return lens


def CookeTriplet():
    lens = optic.Optic()

    lens.surface_group.surfaces = []

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=22.01359, thickness=3.25896,
                     material='SK16')
    lens.add_surface(index=2, radius=-435.76044, thickness=6.00755)
    lens.add_surface(index=3, radius=-22.21328, thickness=0.99997,
                     material=('F2', 'schott'))
    lens.add_surface(index=4, radius=20.29192, thickness=4.75041, is_stop=True)
    lens.add_surface(index=5, radius=79.68360, thickness=2.95208,
                     material='SK16')
    lens.add_surface(index=6, radius=-18.39533, thickness=42.20778)
    lens.add_surface(index=7)

    # add aperture
    lens.set_aperture(aperture_type='EPD', value=10)

    # add field
    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_field(y=14)
    lens.add_field(y=20)

    # add wavelength
    lens.add_wavelength(value=0.48)
    lens.add_wavelength(value=0.55, is_primary=True)
    lens.add_wavelength(value=0.65)

    lens.update_paraxial()

    return lens


def doublelens():
    lens = optic.Optic()

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=56.20238, thickness=8.75,
                     material='N-SSK2')
    lens.add_surface(index=2, radius=152.28580, thickness=0.5)
    lens.add_surface(index=3, radius=37.68262, thickness=12.5, material='N-SK2')
    lens.add_surface(index=4, radius=np.inf, thickness=3.8,
                     material=('F5', 'schott'))
    lens.add_surface(index=5, radius=24.23130, thickness=16.369445)
    lens.add_surface(index=6, radius=np.inf, thickness=13.747957, is_stop=True)
    lens.add_surface(index=7, radius=-28.37731, thickness=3.8,
                     material=('F5', 'schott'))
    lens.add_surface(index=8, radius=np.inf, thickness=11, material='N-SK16')
    lens.add_surface(index=9, radius=-37.92546, thickness=0.5)
    lens.add_surface(index=10, radius=177.41176, thickness=7,
                     material='N-SK16')
    lens.add_surface(index=11, radius=-79.41143, thickness=61.487536)
    lens.add_surface(index=12)

    # add aperture
    lens.set_aperture(aperture_type='imageFNO', value=5)
    lens.set_field_type(field_type='angle')

    # add field
    lens.field_type = 'angle'
    lens.add_field(y=0)
    lens.add_field(y=10)
    lens.add_field(y=14)

    # add wavelength
    lens.add_wavelength(value=0.4861)
    lens.add_wavelength(value=0.5876, is_primary=True)
    lens.add_wavelength(value=0.6563)

    lens.update_paraxial()

    return lens
