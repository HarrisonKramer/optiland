import numpy as np

from optiland import optic, physical_apertures


class HubbleTelescope(optic.Optic):
    """Hubble Space Telescope

    Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 200
    """

    def __init__(self):
        super().__init__()

        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(index=1, thickness=4910.01016)

        obscuration = physical_apertures.RadialAperture(r_max=np.inf, r_min=177.80035)

        self.add_surface(
            index=2,
            radius=-11040.02286,
            thickness=-4910.01016,
            material="mirror",
            is_stop=True,
            conic=-1.001152,
            aperture=obscuration,
        )
        self.add_surface(
            index=3,
            radius=-1349.31166,
            thickness=6365.20955,
            material="mirror",
            conic=-1.483014,
        )
        self.add_surface(index=4, radius=-635.38227)

        self.set_aperture(aperture_type="EPD", value=2400)

        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=0.15)

        self.add_wavelength(value=0.55, is_primary=True)

        self.update_paraxial()
