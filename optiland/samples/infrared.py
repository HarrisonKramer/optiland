import numpy as np

from optiland import materials, optic


class InfraredTriplet(optic.Optic):
    """Infrared Air-Spaced Triplet

    Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 54
    """

    def __init__(self):
        super().__init__()

        germanium = materials.IdealMaterial(n=4.002)

        # https://www.spectral-systems.com/technical-data-sheet/znse-zinc-selenide
        ZnSe = materials.IdealMaterial(n=2.4028)

        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(
            index=1,
            radius=10.4578,
            thickness=0.5901,
            material=germanium,
            is_stop=True,
        )
        self.add_surface(index=2, radius=14.1079, thickness=4.3909)
        self.add_surface(index=3, radius=-15.8842, thickness=0.59, material=ZnSe)
        self.add_surface(index=4, radius=-18.2105, thickness=5.6218)
        self.add_surface(index=5, radius=2.5319, thickness=0.3918, material=germanium)
        self.add_surface(index=6, radius=2.4308, thickness=1.3065)
        self.add_surface(index=7)

        self.set_aperture(aperture_type="imageFNO", value=2)

        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=2.8)
        self.add_field(y=4)

        self.add_wavelength(value=10.6, is_primary=True)

        self.update_paraxial()


class InfraredTripletF4(optic.Optic):
    """4-in. IR Triplet, f/4

    Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 57
    """

    def __init__(self):
        super().__init__()

        # refractive index at 4.2 µm
        germanium = materials.IdealMaterial(n=4.002)
        silicon = materials.IdealMaterial(n=3.4222)

        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(index=1, radius=2.0721, thickness=0.1340, material=silicon)
        self.add_surface(index=2, radius=3.5488, thickness=0.2392)
        self.add_surface(index=3, thickness=0.6105, is_stop=True)
        self.add_surface(index=4, radius=13.7583, thickness=0.1, material=germanium)
        self.add_surface(index=5, radius=1.7491, thickness=0.8768)
        self.add_surface(index=6, thickness=0.1462, material=silicon)
        self.add_surface(index=7, radius=-3.5850, thickness=2.8386)
        self.add_surface(index=8)

        self.set_aperture(aperture_type="imageFNO", value=4)

        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=2.45)
        self.add_field(y=3.5)

        self.add_wavelength(value=4.2, is_primary=True)

        self.update_paraxial()
