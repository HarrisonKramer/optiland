import numpy as np
from optiland import optic


def eyepiece_Erfle():
    """Based on USP1479229, Erfle, Heinrich"""
    lens = optic.Optic()

    lens.surface_group.surfaces = []

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=np.inf, thickness=15.224, is_stop=True)
    lens.add_surface(index=2, radius=269.0, thickness=25.1, material='L-BSL7')
    lens.add_surface(index=3, radius=-125.9, thickness=36.5)
    lens.add_surface(index=4, radius=93.6, thickness=18.5, material='N-BAK2')
    lens.add_surface(index=5, radius=-93.6, thickness=4.1, material='N-F2')
    lens.add_surface(index=6, radius=2550.0, thickness=0.19)
    lens.add_surface(index=7, radius=93.6, thickness=18.5, material='N-BAK2')
    lens.add_surface(index=8, radius=-93.6, thickness=4.1, material='N-F2')
    lens.add_surface(index=9, radius=2550.0, thickness=32.685)
    lens.add_surface(index=10)

    lens.set_aperture(aperture_type='EPD', value=4.0)

    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_field(y=14)
    lens.add_field(y=20)

    lens.add_wavelength(value=0.4861)
    lens.add_wavelength(value=0.5876, is_primary=True)
    lens.add_wavelength(value=0.6563)

    lens.update_paraxial()

    return lens
