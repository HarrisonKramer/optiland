# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 16:12:58 2025

@author: Matteo Taccola

Use of grating surface
"""

from optiland import optic
import numpy as np

lens = optic.Optic()

lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
lens.add_surface(index=1, radius=np.inf, thickness=10)
lens.add_surface(
    index=2, radius=np.inf, thickness=5, material="N-BK7"
)
lens.add_surface(index=3, radius=-100, thickness=30, surface_type="grating", grating_order = 1, grating_period = 5.0, is_stop=True)
lens.add_surface(index=4)

# add aperture
lens.set_aperture(aperture_type="EPD", value=15)

# add field
lens.set_field_type(field_type="angle")
lens.add_field(y=0)
lens.add_field(y=10)

# add wavelength
lens.add_wavelength(value=0.587, is_primary=True)

lens.draw(
    wavelengths=[0.48613270, 0.587561806, 0.65627250],
    figsize=(16, 4),
    num_rays=5,
)

#trace few rays and compare with zemax results
ray_c0 = lens.trace_generic(Hx=0, Hy=0, Px=0, Py=0, wavelength=0.587)
zmx_c0_l,zmx_c0_m,zmx_c0_n = 0,-0.1174,0.9930847094
ray_c1 = lens.trace_generic(Hx=0, Hy=1, Px=0, Py=0, wavelength=0.587)
zmx_c1_l,zmx_c1_m,zmx_c1_n = 0,0.0562738133,0.9984153734
ray_m0 = lens.trace_generic(Hx=0, Hy=0, Px=0, Py=1, wavelength=0.587)
zmx_m0_l,zmx_m0_m,zmx_m0_n = 0,-0.1572064475,0.9875657613
ray_m1 = lens.trace_generic(Hx=0, Hy=1, Px=0, Py=1, wavelength=0.587)
zmx_m1_l,zmx_m1_m,zmx_m1_n = 0,0.0183044729,0.9998324591

print(ray_c0.L[0],ray_c0.M[0],ray_c0.N[0])
print(ray_c1.L[0],ray_c1.M[0],ray_c1.N[0])
print(ray_m0.L[0],ray_m0.M[0],ray_m0.N[0])
print(ray_m1.L[0],ray_m1.M[0],ray_m1.N[0])