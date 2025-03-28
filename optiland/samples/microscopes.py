import numpy as np

from optiland import materials, optic, physical_apertures


class Objective60x(optic.Optic):
    """define a microscopescope objective"""

    def __init__(self):
        super().__init__()

        self.add_surface(index=0, thickness=np.inf, radius=np.inf)
        self.add_surface(index=1, thickness=64.9, radius=553.260, material="N-FK51")
        self.add_surface(index=2, thickness=4.4, radius=-247.644)
        self.add_surface(index=3, thickness=59.4, radius=115.162, material="J-LLF2")
        self.add_surface(index=4, thickness=17.6, radius=57.131)
        self.add_surface(index=5, thickness=17.6, is_stop=True)
        self.add_surface(
            index=6,
            thickness=74.8,
            radius=-57.646,
            material=("SF5", "schott"),
        )
        self.add_surface(index=7, thickness=77.0, radius=196.614, material="N-FK51")
        self.add_surface(index=8, thickness=4.4, radius=-129.243)
        self.add_surface(index=9, thickness=15.4, radius=2062.370, material="N-KZFS4")
        self.add_surface(
            index=10,
            thickness=48.4,
            radius=203.781,
            material="LITHOTEC-CAF2",
        )
        self.add_surface(index=11, thickness=4.4, radius=-224.003)
        self.add_surface(
            index=12,
            thickness=35.2,
            radius=219.864,
            material="LITHOTEC-CAF2",
        )
        self.add_surface(index=13, thickness=4.4, radius=793.3)
        self.add_surface(index=14, thickness=26.4, radius=349.260, material="N-FK51")
        self.add_surface(index=15, thickness=4.4, radius=-401.950)
        self.add_surface(index=16, thickness=39.6, radius=91.992, material="N-SK11")
        self.add_surface(index=17, thickness=96.189, radius=176.0)
        self.add_surface(index=18)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=0.9)

        # add field
        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=0.7)
        self.add_field(y=1)

        # add wavelength
        self.add_wavelength(value=0.4861)
        self.add_wavelength(value=0.5876, is_primary=True)
        self.add_wavelength(value=0.6563)

        self.update_paraxial()


class Microscope20x(optic.Optic):
    """20x Microscope Objective

    Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 135
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(index=1, radius=-0.2352, thickness=0.0941, material="N-SK16")
        self.add_surface(
            index=2,
            radius=-0.1968,
            thickness=0.0413,
            material=("SF4", "schott"),
        )
        self.add_surface(index=3, radius=-0.3251, thickness=0.01)
        self.add_surface(index=4, radius=0.5837, thickness=0.1115, material="N-SK16")
        self.add_surface(index=5, radius=-0.9401, thickness=0.01)
        self.add_surface(index=6, radius=np.inf, thickness=0.2236, is_stop=True)
        self.add_surface(index=7, radius=0.2077, thickness=0.2, material="N-SK16")
        self.add_surface(
            index=8,
            radius=-0.1686,
            thickness=0.025,
            material=("SF4", "schott"),
        )
        self.add_surface(index=9, radius=0.4108, thickness=0.0965)
        self.add_surface(index=10, radius=np.inf, thickness=0.007, material="N-K5")
        self.add_surface(index=11)

        # add aperture
        self.set_aperture(aperture_type="EPD", value=0.317961)

        # add field
        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=0.7)
        self.add_field(y=1)

        # add wavelength
        self.add_wavelength(value=0.48613270)
        self.add_wavelength(value=0.58756180, is_primary=True)
        self.add_wavelength(value=0.65627250)

        self.update_paraxial()
        self.image_solve()

        # scale from inches to mm
        self.scale_system(25.4)


class UVReflectingMicroscope(optic.Optic):
    """53x UV Reflecting Objective

    Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 139
    """

    def __init__(self):
        super().__init__()

        silica = materials.IdealMaterial(n=1.4980)
        CaF2 = materials.IdealMaterial(n=1.4610)

        self.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.add_surface(index=1, radius=1.72300, thickness=0.0350, material=silica)
        self.add_surface(index=2, radius=2.90631, thickness=0.9731)
        self.add_surface(
            index=3,
            radius=0.17783,
            thickness=-0.4995,
            material="mirror",
            is_stop=True,
        )

        obscuration = physical_apertures.RadialAperture(r_max=np.inf, r_min=0.15)
        self.add_surface(
            index=4,
            radius=0.72913,
            thickness=0.5776,
            material="mirror",
            aperture=obscuration,
        )

        self.add_surface(index=5, radius=2.66996, thickness=0.0427, material=CaF2)
        self.add_surface(index=6, radius=0.48467, thickness=0.005)
        self.add_surface(index=7, radius=0.23767, thickness=0.1861, material=CaF2)
        self.add_surface(index=8, radius=8.64894, thickness=0.005)
        self.add_surface(index=9, radius=7.25758, thickness=0.0588, material=silica)
        self.add_surface(index=10, radius=0.44435, thickness=0.0771)
        self.add_surface(index=11, thickness=0.0071, material=silica)
        self.add_surface(index=12)

        self.set_aperture(aperture_type="imageFNO", value=0.7)

        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_field(y=2.8)
        self.add_field(y=4)

        self.add_wavelength(value=0.27, is_primary=True)

        self.update_paraxial()

        # scale from inches to mm
        self.scale_system(25.4)
