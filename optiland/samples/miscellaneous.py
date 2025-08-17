from __future__ import annotations

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic

__all__ = ["NavarroWideAngleEye"]


class NavarroWideAngleEye(Optic):
    """Navarro wide-angle schematic eye for 543 nm light.

    Schematic eye model proposed by `Escudero-Sanz and Navarro`_ (1999) using the
    refractive indices for light at 543 nm.

    .. _Escudero-Sanz and Navarro:
       https://doi.org/10.1364/JOSAA.16.001881
    """

    materials = {
        "cornea": IdealMaterial(1.3777),
        "aqueous_humor": IdealMaterial(1.3391),
        "lens": IdealMaterial(1.4222),
        "vitreous_humor": IdealMaterial(1.3377),
    }

    def __init__(self):
        super().__init__()

        self.add_surface(index=0, comment="object", radius=be.inf, thickness=be.inf)
        self.add_surface(
            index=1,
            comment="cornea front",
            radius=7.72,
            thickness=0.55,
            conic=-0.26,
            material=self.materials["cornea"],
        )
        self.add_surface(
            index=2,
            comment="cornea back",
            radius=6.5,
            thickness=3.05,
            conic=0,
            material=self.materials["aqueous_humor"],
        )
        self.add_surface(
            index=3,
            comment="pupil",
            radius=be.inf,
            thickness=0.0,
            is_stop=True,
            material=self.materials["aqueous_humor"],
        )
        self.add_surface(
            index=4,
            comment="lens front",
            radius=10.2,
            thickness=4,
            conic=-3.1316,
            material=self.materials["lens"],
        )
        self.add_surface(
            index=5,
            comment="lens back",
            radius=-6,
            thickness=16.3203,
            conic=-1,
            material=self.materials["vitreous_humor"],
        )
        self.add_surface(
            index=6,
            comment="retina",
            radius=-12,
            material=self.materials["vitreous_humor"],
        )

        self.set_aperture(aperture_type="float_by_stop_size", value=3.0)

        self.set_field_type(field_type="angle")
        self.add_field(0)
        self.add_field(15)
        self.add_field(30)
        self.add_field(45)
        self.add_field(60)

        self.add_wavelength(0.543)
