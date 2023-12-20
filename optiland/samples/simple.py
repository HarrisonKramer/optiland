import numpy as np
from optiland import optic


def Edmund_49_847():
    '''define Edmund optics 49-847'''
    singlet = optic.Optic()

    # add surfaces
    singlet.add_surface(index=0, radius=np.inf, thickness=np.inf)
    singlet.add_surface(index=1, thickness=7, radius=19.93,
                        is_stop=True, material='N-SF11')
    singlet.add_surface(index=2, thickness=21.48)
    singlet.add_surface(index=3)

    # add aperture
    singlet.set_aperture(aperture_type='EPD', value=25.4)

    # add field
    singlet.set_field_type(field_type='angle')
    singlet.add_field(number=0, x=0, y=0)
    singlet.add_field(number=1, x=0, y=10)
    singlet.add_field(number=2, x=0, y=14)

    # add wavelength
    singlet.add_wavelength(number=0, value=0.4861)
    singlet.add_wavelength(number=1, value=0.5876, is_primary=True)
    singlet.add_wavelength(number=2, value=0.6563)

    singlet.update_paraxial()

    return singlet
