import numpy as np
from optiland import optic


def Edmund_49_847():
    """Edmund optics 49-847"""
    lens = optic.Optic()

    # add surfaces
    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, thickness=7, radius=19.93, is_stop=True,
                     material='N-SF11')
    lens.add_surface(index=2, thickness=21.48)
    lens.add_surface(index=3)

    # add aperture
    lens.set_aperture(aperture_type='EPD', value=25.4)

    # add field
    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_field(y=10)
    lens.add_field(y=14)

    # add wavelength
    lens.add_wavelength(value=0.48613270)
    lens.add_wavelength(value=0.58756180, is_primary=True)
    lens.add_wavelength(value=0.65627250)

    lens.update_paraxial()

    return lens


def SingletStopSurf2():
    """A simple singlet with the stop on surface 2"""
    lens = optic.Optic()

    # add surfaces
    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, thickness=10.0, radius=63.73364157,
                     material='LAC9')
    lens.add_surface(index=2, thickness=92.73834630, radius=653.29392320,
                     is_stop=True)
    lens.add_surface(index=3)

    # add aperture
    lens.set_aperture(aperture_type='EPD', value=25.0)

    # add field
    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_field(y=3.5)
    lens.add_field(y=5)

    # add wavelength
    lens.add_wavelength(value=0.48613270)
    lens.add_wavelength(value=0.58756180, is_primary=True)
    lens.add_wavelength(value=0.65627250)

    lens.update_paraxial()

    return lens
