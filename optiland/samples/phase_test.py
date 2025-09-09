# Defines simple sample optical systems.
import optiland.backend as be
from optiland import optic
#from optiland.phase.grating import GratingPhase
from optiland.phase.radial import RadialPhase
import math as mth
import numpy as np
from optiland.materials import AbbeMaterial, Material
from optiland import wavefront
import matplotlib.pyplot as plt



class AsphericSingletMirror(optic.Optic):
    """An aspheric singlet lens."""

    def __init__(self):
        super().__init__()
        # add surfaces
        self.add_surface(index=0, radius=be.inf, thickness=be.inf)
        # self.add_surface(index=1, radius=be.inf, thickness=1, material = "N-BK7")
        self.add_surface(index=1, radius=be.inf, thickness=1, material = "air")
        self.add_surface(
            index=2,
            thickness=-100,
            radius= 500000000000,
            is_stop=True,
            material= "mirror",
            surface_type="standard",
            #phase_type = GratingPhase(A = 1, order = -1, eff = 'ideal')
            phase_type = RadialPhase(coef = [0.071, -.0005], order = -1, eff = 'ideal')
        )
        self.add_surface(index=3, thickness=0)
        self.add_surface(index=4)

        # add aperture
        self.set_aperture(aperture_type="EPD", value=20.0)

        # add field
        self.set_field_type(field_type="angle")
        self.add_field(y=0)

        # add wavelength
        self.add_wavelength(value=0.530, is_primary=True)

class AsphericSinglet(optic.Optic):
    """An aspheric singlet lens."""

    def __init__(self):
        super().__init__()
        # add surfaces
        self.add_surface(index=0, radius=be.inf, thickness=be.inf)
        self.add_surface(index=1, radius=be.inf, thickness=1, material = "N-BK7")
        # self.add_surface(index=1, radius=be.inf, thickness=1, material = "air")
        self.add_surface(
            index=2,
            thickness=100,
            radius= 500000000000,
            is_stop=True,
            material= "air",
            surface_type="standard",
            phase_type = GratingPhase(A = 1, order = -1, eff = 'ideal')
        )
        self.add_surface(index=3, thickness=0)
        self.add_surface(index=4)

        # add aperture
        self.set_aperture(aperture_type="EPD", value=20.0)

        # add field
        self.set_field_type(field_type="angle")
        self.add_field(y=0)

        # add wavelength
        self.add_wavelength(value=0.530, is_primary=True)
        
lens = AsphericSingletMirror()
# You may try a different lens here
#lens = AsphericSinglet()
bk7=Material("N-Bk7")
print(bk7.n(0.530))
#rays = lens.trace(Hx=0, Hy=0, wavelength=0.53, num_rays=1 )
rayData = lens.trace_generic(Hx=0, Hy=0,Px = 0, Py = 1, wavelength=0.53)


#rayDataFull = lens.trace(Hx = 0, Hy = 0, wavelength = 0.530, num_rays=1, distribution="line_y")
num_surfaces = lens.surface_group.num_surfaces
# take intersection points on last surface only
# print(lens.surface_group.y[:,:],lens.surface_group.z[:,:], lens.surface_group.M[:,:],lens.surface_group.N[:,:])
#x_image = lens.surface_group.x[num_surfaces - 1, :]
#y_image = lens.surface_group.y[num_surfaces - 1, :]
#print(rayDataFull)
# print(rayData)
# print(mth.atan(rayData.M/rayData.N)*180/mth.pi)
lens.draw3D(distribution="line_y")
#lens.draw()
#opd = wavefront.OPD(lens, field=(0, 0), wavelength=0.530)
#opd.view(projection="3d", num_points=128)
#plt.show()
#rays = lens.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=1024, distribution="random")