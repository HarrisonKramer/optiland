import numpy as np

from optiland import optic


class EyepieceErfle(optic.Optic):
    """Based on USP1479229, Erfle, Heinrich"""

    def __init__(self):
        super().__init__()

        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(index=1, radius=np.inf, thickness=15.224, is_stop=True)
        self.add_surface(index=2, radius=269.0, thickness=25.1, material="L-BSL7")
        self.add_surface(index=3, radius=-125.9, thickness=36.5)
        self.add_surface(index=4, radius=93.6, thickness=18.5, material="N-BAK2")
        self.add_surface(index=5, radius=-93.6, thickness=4.1, material="N-F2")
        self.add_surface(index=6, radius=2550.0, thickness=0.19)
        self.add_surface(index=7, radius=93.6, thickness=18.5, material="N-BAK2")
        self.add_surface(index=8, radius=-93.6, thickness=4.1, material="N-F2")
        self.add_surface(index=9, radius=2550.0, thickness=32.685)
        self.add_surface(index=10)

        self.set_aperture(aperture_type="EPD", value=4.0)

        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=14)
        self.add_field(y=20)

        self.add_wavelength(value=0.4861)
        self.add_wavelength(value=0.5876, is_primary=True)
        self.add_wavelength(value=0.6563)

        self.update_paraxial()
