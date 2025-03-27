import numpy as np

from optiland import materials, optic


class UVProjectionLens(optic.Optic):
    """UV projection lens for 248 nm lithography.

    Based on U.S. Patent #5831776
    """

    def __init__(self):
        super().__init__()

        # We define SiO2 with the index of refraction at 248 nm
        SiO2 = materials.IdealMaterial(n=1.5084, k=0)

        # Define all surfaces
        self.add_surface(index=0, radius=np.inf, thickness=110.85883544)
        self.add_surface(index=1, radius=-737.7847, thickness=27.484, material=SiO2)
        self.add_surface(index=2, radius=-235.2891, thickness=0.916)
        self.add_surface(index=3, radius=211.1786, thickness=36.646, material=SiO2)
        self.add_surface(index=4, radius=-461.3986, thickness=0.916)
        self.add_surface(index=5, radius=412.6778, thickness=21.071, material=SiO2)
        self.add_surface(index=6, radius=160.5391, thickness=16.197)
        self.add_surface(index=7, radius=-604.1283, thickness=7.215, material=SiO2)
        self.add_surface(index=8, radius=218.1877, thickness=23.941)
        self.add_surface(index=9, radius=-3586.063, thickness=11.978, material=SiO2)
        self.add_surface(index=10, radius=251.8168, thickness=47.506)
        self.add_surface(index=11, radius=-85.2817, thickness=11.961, material=SiO2)
        self.add_surface(index=12, radius=584.8597, thickness=9.968)
        self.add_surface(index=13, radius=4074.801, thickness=35.291, material=SiO2)
        self.add_surface(index=14, radius=-162.0185, thickness=0.923)
        self.add_surface(index=15, radius=629.544, thickness=41.227, material=SiO2)
        self.add_surface(index=16, radius=-226.7397, thickness=0.916)
        self.add_surface(index=17, radius=522.2739, thickness=27.842, material=SiO2)
        self.add_surface(index=18, radius=-582.424, thickness=0.916)
        self.add_surface(index=19, radius=423.729, thickness=22.904, material=SiO2)
        self.add_surface(index=20, radius=-1385.36, thickness=0.916, is_stop=True)
        self.add_surface(index=21, radius=212.039, thickness=33.646, material=SiO2)
        self.add_surface(index=22, radius=802.3695, thickness=55.304)
        self.add_surface(index=23, radius=-776.5697, thickness=8.703, material=SiO2)
        self.add_surface(index=24, radius=106.1728, thickness=24.09)
        self.add_surface(index=25, radius=-200.683, thickness=11.452, material=SiO2)
        self.add_surface(index=26, radius=311.8264, thickness=59.54)
        self.add_surface(index=27, radius=-77.2276, thickness=11.772, material=SiO2)
        self.add_surface(index=28, radius=2317.8032, thickness=11.862)
        self.add_surface(index=29, radius=-290.8859, thickness=22.904, material=SiO2)
        self.add_surface(index=30, radius=-148.3577, thickness=1.373)
        self.add_surface(index=31, radius=-5658.5043, thickness=41.227, material=SiO2)
        self.add_surface(index=32, radius=-151.9858, thickness=0.916)
        self.add_surface(index=33, radius=678.1005, thickness=32.981, material=SiO2)
        self.add_surface(index=34, radius=-358.554, thickness=0.916)
        self.add_surface(index=35, radius=264.2734, thickness=32.814, material=SiO2)
        self.add_surface(index=36, radius=2309.6884, thickness=0.916)
        self.add_surface(index=37, radius=171.2681, thickness=29.015, material=SiO2)
        self.add_surface(index=38, radius=364.7765, thickness=0.918)
        self.add_surface(index=39, radius=113.37, thickness=76.259, material=SiO2)
        self.add_surface(index=40, radius=78.6982, thickness=54.304)
        self.add_surface(index=41, radius=49.5443, thickness=18.65, material=SiO2)
        self.add_surface(index=42, radius=109.8136, thickness=13.07647896)
        self.add_surface(index=43, radius=np.inf)

        # Define the aperture
        self.set_aperture(aperture_type="objectNA", value=0.133)

        # Define the field
        self.set_field_type(field_type="object_height")
        self.add_field(y=0)
        self.add_field(y=32)
        self.add_field(y=48)

        # Define the wavelength
        self.add_wavelength(value=0.248, is_primary=True)

        # Specify that the self is object-space telecentric
        self.obj_space_telecentric = True

        # Move last surface to the paraxial image plane
        self.image_solve()
