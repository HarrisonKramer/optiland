# Defines simple sample optical systems.
import optiland.backend as be
from optiland import optic
from optiland.phase.grating import GratingPhase
import math as mth
import numpy as np
from optiland.materials import AbbeMaterial, Material




class AsphericSinglet(optic.Optic):
    """An aspheric singlet lens."""

    def __init__(self):
        super().__init__()
        # add surfaces
        self.add_surface(index=0, radius=be.inf, thickness=be.inf)
        self.add_surface(index=1, radius=be.inf, thickness=1, material = "mirror")
        self.add_surface(
            index=2,
            thickness=-100,
            radius= 50,
            is_stop=True,
            material= "air",
            surface_type="standard",
            phase_type = GratingPhase(A = 1, order = 1)
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
        
lens = AsphericSinglet()
bk7=Material("N-Bk7")
print(bk7.n(0.530))
#rays = lens.trace(Hx=0, Hy=0, wavelength=0.53, num_rays=1 )
rayData = lens.trace_generic(Hx=0, Hy=0,Px = 0, Py = 1, wavelength=0.53)


#rayDataFull = lens.trace(Hx = 0, Hy = 0, wavelength = 0.530, num_rays=1, distribution="line_y")
num_surfaces = lens.surface_group.num_surfaces
# take intersection points on last surface only
print(lens.surface_group.y[:,:],lens.surface_group.z[:,:], lens.surface_group.M[:,:],lens.surface_group.N[:,:])
#x_image = lens.surface_group.x[num_surfaces - 1, :]
#y_image = lens.surface_group.y[num_surfaces - 1, :]
#print(rayDataFull)
print(rayData)
print(mth.atan(rayData.M/rayData.N)*180/mth.pi)
lens.draw3D(distribution="line_y")
#lens.draw()

#rays = lens.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=1024, distribution="random")