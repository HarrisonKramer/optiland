import numpy as np

from optiland import optic


class Edmund_49_847(optic.Optic):
    """Edmund optics 49-847"""

    def __init__(self):
        super().__init__()

        # add surfaces
        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(
            index=1,
            thickness=7,
            radius=19.93,
            is_stop=True,
            material="N-SF11",
        )
        self.add_surface(index=2, thickness=21.48)
        self.add_surface(index=3)

        # add aperture
        self.set_aperture(aperture_type="EPD", value=25.4)

        # add field
        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=10)
        self.add_field(y=14)

        # add wavelength
        self.add_wavelength(value=0.48613270)
        self.add_wavelength(value=0.58756180, is_primary=True)
        self.add_wavelength(value=0.65627250)

        self.update_paraxial()


class SingletStopSurf2(optic.Optic):
    """A simple singlet with the stop on surface 2"""

    def __init__(self):
        super().__init__()

        # add surfaces
        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(index=1, thickness=10.0, radius=63.73364157, material="LAC9")
        self.add_surface(
            index=2,
            thickness=92.73834630,
            radius=653.29392320,
            is_stop=True,
        )
        self.add_surface(index=3)

        # add aperture
        self.set_aperture(aperture_type="EPD", value=25.0)

        # add field
        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=3.5)
        self.add_field(y=5)

        # add wavelength
        self.add_wavelength(value=0.48613270)
        self.add_wavelength(value=0.58756180, is_primary=True)
        self.add_wavelength(value=0.65627250)

        self.update_paraxial()


class TelescopeDoublet(optic.Optic):
    """Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 44"""

    def __init__(self):
        super().__init__()

        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(
            index=1,
            radius=29.32908,
            thickness=0.7,
            material="N-BK7",
            is_stop=True,
        )
        self.add_surface(index=2, radius=-20.06842, thickness=0.032)
        self.add_surface(
            index=3,
            radius=-20.08770,
            thickness=0.5780,
            material=("SF2", "schott"),
        )
        self.add_surface(index=4, radius=-66.54774, thickness=47.3562)
        self.add_surface(index=5)

        self.set_aperture(aperture_type="imageFNO", value=8.0)

        self.set_field_type(field_type="angle")
        self.add_field(y=0.0)
        self.add_field(y=0.7)
        self.add_field(y=1.0)

        self.add_wavelength(value=0.4861)
        self.add_wavelength(value=0.5876, is_primary=True)
        self.add_wavelength(value=0.6563)

        self.update_paraxial()
        self.image_solve()


class CementedAchromat(optic.Optic):
    """Cemented Achromatic Doublet

    Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 45
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(
            index=1,
            radius=12.38401,
            thickness=0.4340,
            is_stop=True,
            material="N-BAK1",
        )
        self.add_surface(
            index=2,
            radius=-7.94140,
            thickness=0.3210,
            material=("SF2", "schott"),
        )
        self.add_surface(index=3, radius=-48.44396, thickness=19.6059)
        self.add_surface(index=4)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=6)

        # add field
        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=3.5)
        self.add_field(y=5)

        # add wavelength
        self.add_wavelength(value=0.48613270)
        self.add_wavelength(value=0.58756180, is_primary=True)
        self.add_wavelength(value=0.65627250)

        self.update_paraxial()
        self.image_solve()


class AsphericSinglet(optic.Optic):
    """Aspheric singlet"""

    def __init__(self):
        super().__init__()

        # add surfaces
        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(
            index=1,
            thickness=7,
            radius=20.0,
            is_stop=True,
            material="N-SF11",
            surface_type="even_asphere",
            conic=0.0,
            coefficients=[-2.248851e-4, -4.690412e-6, -6.404376e-8],
        )
        self.add_surface(index=2, thickness=21.56201105)
        self.add_surface(index=3)

        # add aperture
        self.set_aperture(aperture_type="EPD", value=20.0)

        # add field
        self.set_field_type(field_type="angle")
        self.add_field(y=0)

        # add wavelength
        self.add_wavelength(value=0.587, is_primary=True)

        self.update_paraxial()
