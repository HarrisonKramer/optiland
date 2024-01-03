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
