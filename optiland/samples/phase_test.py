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
        a1 = -0.01
        print(a1)
        self.add_surface(index=0, radius=be.inf, thickness=be.inf)
        # self.add_surface(index=1, radius=be.inf, thickness=1, material = "N-BK7")
        self.add_surface(index=1, radius=be.inf, thickness=0, material = "air")
        self.add_surface(
            index=2,
            thickness=-50,
            radius= -75,
            is_stop=True,
            material= "mirror",
            surface_type="standard",
            #phase_type = GratingPhase(A = 1, order = -1, eff = 'ideal')
            phase_type = RadialPhase(coef = [a1], order = -1, eff = 'ideal')
        )
        self.add_surface(index=3, thickness=-0)
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
            radius= 50,
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

rayData = lens.trace_generic(Hx=0, Hy=0,Px = 0, Py = 1, wavelength=0.53)
num_surfaces = lens.surface_group.num_surfaces
lens.draw(distribution="line_y",num_rays=11)
plt.show()


def generate_wavefront(z, num_points=32):
    x, y = np.meshgrid(np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points))
    radius = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    values = z.poly(radius, phi)

    values[radius > 1] = 0.0

    # Normalize the values between -1 and 1
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    values = 2 * values - 1

    return values
num_points = 65
A = 10
x, y = np.meshgrid(np.linspace(-1*A, 1*A, num_points), np.linspace(-1*A, 1*A, num_points))
radius = np.sqrt(x**2 + y**2)
#p[radius > 1*A] = 0.0



def plot_phase(values,A, num_points):
    _, ax = plt.subplots(figsize=(7, 5.5))

    x, y = np.meshgrid(np.linspace(-1*A, 1*A, num_points), np.linspace(-1*A, 1*A, num_points))
    radius = np.sqrt(x**2 + y**2)
    p=RadialPhase()
    p.order = -1
    p.coef= [-0.01]
    
    values=m*2*pi/0.53e-3*p.phasefunction( x, y)
    values[radius > 1*A] = 0.0
    im = ax.imshow(np.flipud(values), extent=[-1*A, 1*A, -1*A, 1*A], cmap = "jet")
    #im = ax.plot(np.linspace(-1*A, 1*A, num_points),values[int(33),:])
    

    ax.set_xlabel("Pupil X")
    ax.set_ylabel("Pupil Y")
    plt.colorbar(im)
    plt.show()
    


plot_phase(0.01*2*np.pi/0.53e-3, 10, num_points)

