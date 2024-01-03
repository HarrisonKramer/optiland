import numpy as np
from optiland import optic, physical_apertures


def hubble():
    '''https://lambdares.com/support-posts/hubble-len-the-hubble-space-telescope'''

    lens = optic.Optic()

    lens.surface_group.surfaces = []

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)

    ap = physical_apertures.RadialAperture(r_max=np.inf, r_min=0.155e3)
    lens.add_surface(index=1, thickness=5.0e3, aperture=ap)

    lens.add_surface(index=2, thickness=-4.906071e3, radius=-11.04e3,
                     conic=-1.00229850, is_stop=True, material='mirror',
                     aperture=ap)
    lens.add_surface(index=3, thickness=6.40619954e3, radius=-1.358e3,
                     conic=-1.49686, material='mirror')
    lens.add_surface(index=4, radius=-0.6310792e3)

    lens.set_aperture(aperture_type='EPD', value=2.4e3)

    lens.set_field_type(field_type='angle')
    lens.add_field(y=0.0)
    lens.add_field(y=0.06)
    lens.add_field(y=0.08)

    lens.add_wavelength(value=0.5, is_primary=True)

    lens.update_paraxial()

    return lens
