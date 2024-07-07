import numpy as np
from optiland import optic


class Edmund_49_847(optic.Optic):
    """Edmund optics 49-847"""

    def __init__(self):
        super().__init__()

        # add surfaces
        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(index=1, thickness=7, radius=19.93, is_stop=True,
                         material='N-SF11')
        self.add_surface(index=2, thickness=21.48)
        self.add_surface(index=3)

        # add aperture
        self.set_aperture(aperture_type='EPD', value=25.4)

        # add field
        self.set_field_type(field_type='angle')
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
        self.add_surface(index=1, thickness=10.0, radius=63.73364157,
                         material='LAC9')
        self.add_surface(index=2, thickness=92.73834630, radius=653.29392320,
                         is_stop=True)
        self.add_surface(index=3)

        # add aperture
        self.set_aperture(aperture_type='EPD', value=25.0)

        # add field
        self.set_field_type(field_type='angle')
        self.add_field(y=0)
        self.add_field(y=3.5)
        self.add_field(y=5)

        # add wavelength
        self.add_wavelength(value=0.48613270)
        self.add_wavelength(value=0.58756180, is_primary=True)
        self.add_wavelength(value=0.65627250)

        self.update_paraxial()
