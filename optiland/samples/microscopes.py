import numpy as np
from optiland import optic


def Objective60x():
    '''define a microscopescope objective'''
    microscope = optic.Optic()

    microscope.add_surface(index=0, thickness=np.inf, radius=np.inf)
    microscope.add_surface(index=1, thickness=64.9, radius=553.260, material='N-FK51')
    microscope.add_surface(index=2, thickness=4.4, radius=-247.644)
    microscope.add_surface(index=3, thickness=59.4, radius=115.162, material='J-LLF2')
    microscope.add_surface(index=4, thickness=17.6, radius=57.131)
    microscope.add_surface(index=5, thickness=17.6, is_stop=True)
    microscope.add_surface(index=6, thickness=74.8, radius=-57.646, material=('SF5', 'schott'))
    microscope.add_surface(index=7, thickness=77.0, radius=196.614, material='N-FK51')
    microscope.add_surface(index=8, thickness=4.4, radius=-129.243)
    microscope.add_surface(index=9, thickness=15.4, radius=2062.370, material='N-KZFS4')
    microscope.add_surface(index=10, thickness=48.4, radius=203.781, material='CAF2')
    microscope.add_surface(index=11, thickness=4.4, radius=-224.003)
    microscope.add_surface(index=12, thickness=35.2, radius=219.864, material='CAF2')
    microscope.add_surface(index=13, thickness=4.4, radius=793.3)
    microscope.add_surface(index=14, thickness=26.4, radius=349.260, material='N-FK51')
    microscope.add_surface(index=15, thickness=4.4, radius=-401.950)
    microscope.add_surface(index=16, thickness=39.6, radius=91.992, material='N-SK11')
    microscope.add_surface(index=17, thickness=96.189, radius=176.0)
    microscope.add_surface(index=18)

    # add aperture
    microscope.set_aperture(aperture_type='imageFNO', value=0.9)

    # add field
    microscope.set_field_type(field_type='angle')
    microscope.add_field(y=0)
    microscope.add_field(y=0.7)
    microscope.add_field(y=1)

    # add wavelength
    microscope.add_wavelength(value=0.4861)
    microscope.add_wavelength(value=0.5876, primary=True)
    microscope.add_wavelength(value=0.6563)

    microscope.update_paraxial()

    return microscope
