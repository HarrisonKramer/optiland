import numpy as np
from optiland import optic


def triplet_telescope_objective():
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


def Cooke_triplet():
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


def double_Gauss():
    lens = optic.Optic()

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=56.20238, thickness=8.75,
                     material='N-SSK2')
    lens.add_surface(index=2, radius=152.28580, thickness=0.5)
    lens.add_surface(index=3, radius=37.68262, thickness=12.5,
                     material='N-SK2')
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


def reverse_telephoto():
    """An optimized design with starting point USP 2959100"""
    lens = optic.Optic()

    lens.surface_group.surfaces = []

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=1.69111096, thickness=0.08259680,
                     material='N-SK10')
    lens.add_surface(index=2, radius=0.94414496, thickness=0.8)
    lens.add_surface(index=3, radius=4.32100401, thickness=0.080256,
                     material='SK15')
    lens.add_surface(index=4, radius=1.78117621, thickness=0.5)
    lens.add_surface(index=5, radius=2.64050282, thickness=0.27638160,
                     material='BASF2')
    lens.add_surface(index=6, radius=-3.86177348, thickness=0.1)
    lens.add_surface(index=7, radius=1.05627661, thickness=0.2, material='FK3')
    lens.add_surface(index=8, radius=-4.06933311, thickness=0.2001384)
    lens.add_surface(index=9, radius=np.inf, thickness=0.06688, is_stop=True)
    lens.add_surface(index=10, radius=-2.61246583, thickness=0.064372,
                     material=('SF15', 'hikari'))
    lens.add_surface(index=11, radius=0.99117409, thickness=0.3)
    lens.add_surface(index=12, radius=9.03045960, thickness=0.18743120,
                     material='N-LAK12')
    lens.add_surface(index=13, radius=-1.35680743, thickness=2.35130547)
    lens.add_surface(index=14)

    # add aperture
    lens.set_aperture(aperture_type='EPD', value=0.3)

    # add field
    lens.set_field_type(field_type='angle')
    lens.add_field(y=0)
    lens.add_field(y=21)
    lens.add_field(y=30)

    # add wavelength
    lens.add_wavelength(value=0.4861)
    lens.add_wavelength(value=0.5876, is_primary=True)
    lens.add_wavelength(value=0.6563)

    lens.update_paraxial()

    return lens


def objective_US008879901():
    """An optimized design with starting point US008879901"""
    lens = optic.Optic()
    lens.surface_group.surfaces = []

    lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
    lens.add_surface(index=1, radius=47.07125235, thickness=5.29811826, material='N-LAF32')
    lens.add_surface(index=2, radius=184.28171667, thickness=0.6)
    lens.add_surface(index=3, radius=29.92177645, thickness=7.13654863, material='H-ZLAF52A')
    lens.add_surface(index=4, radius=50.4992638, thickness=2.0)
    lens.add_surface(index=5, radius=60.5004845, thickness=0.99941671, material='E-SF1')
    lens.add_surface(index=6, radius=17.72638376, thickness=9.9)
    lens.add_surface(index=7, radius=np.inf, thickness=8.7, is_stop=True)
    lens.add_surface(index=8, radius=-17.49862241, thickness=1.29934579, material=('SF4', 'hikari'))
    lens.add_surface(index=9, radius=1000.00000019, thickness=8.44325264, material='M-TAF1')
    lens.add_surface(index=10, radius=-28.00122422, thickness=0.1)
    lens.add_surface(index=11, radius=-141.99976777, thickness=6.79950254, material='M-TAF1')
    lens.add_surface(index=12, radius=-35.94103045, thickness=0.516)
    lens.add_surface(index=13, radius=92.00034667, thickness=3.29901361, material='Q-LAFPH1S')
    lens.add_surface(index=14, radius=-277.85210888, thickness=2.13)
    lens.add_surface(index=15, radius=-157.24588662, thickness=1.29980422, material='S-FSL5')
    lens.add_surface(index=16, radius=740.47397742, thickness=0.25)
    lens.add_surface(index=17, radius=19.91929498, thickness=5.59345688, material='J-LASF015')
    lens.add_surface(index=18, radius=36.48852623, thickness=0.574)
    lens.add_surface(index=19, radius=45.97532235, thickness=1.00045731, material='E-SF1')
    lens.add_surface(index=20, radius=16.39521847, thickness=2.951)
    lens.add_surface(index=21, radius=33.86131631, thickness=3.22444231, material='H-LAK52')
    lens.add_surface(index=22, radius=np.inf, thickness=8.0)
    lens.add_surface(index=23, radius=np.inf, thickness=4.0, material='H-LAK52')
    lens.add_surface(index=24, radius=np.inf, thickness=3.15317838)
    lens.add_surface(index=25)

    lens.set_aperture(aperture_type='imageFNO', value=2.0)

    lens.set_field_type(field_type='angle')
    lens.add_field(0.0)
    lens.add_field(7.574)
    lens.add_field(10.82)

    lens.add_wavelength(value=0.4861327)
    lens.add_wavelength(value=0.5875618, is_primary=True)
    lens.add_wavelength(value=0.6562725)

    lens.update_paraxial()

    return lens
