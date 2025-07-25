# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 16:12:58 2025

@author: Matteo Taccola

Use of grating surface
"""

from optiland import optic
from optiland import analysis
import numpy as np

lens = optic.Optic()

lens.add_surface(index=0, radius=np.inf, thickness = 18.0)
lens.add_surface(index=1, radius=135.903, thickness=33.53, material="N-BK7",aperture = 60)
lens.add_surface(index=2, radius=-67.699, thickness=60.668, aperture = 60)
lens.add_surface(index=3, radius=-51.467, thickness=8.413, material="N-BK7", aperture = 50)
lens.add_surface(index=4, radius=-65.577, thickness=39.389, aperture = 50)
lens.add_surface(
    index=5, 
    radius=-154.020, 
    thickness=-39.389, 
    material = "mirror", 
    surface_type="grating", 
    grating_order = -1, 
    grating_period = 10.0, 
    is_stop=True,
    #aperture=40.0
    )
lens.add_surface(index=6, radius=-65.577, thickness=-8.413, material="N-BK7", aperture = 50)
lens.add_surface(index=7, radius=-51.467, thickness=-60.668, aperture = 50)
lens.add_surface(index=8, radius=-67.699, thickness=-33.53, material="N-BK7",aperture = 60)
lens.add_surface(index=9, radius=135.903, thickness = -18.0,aperture = 60)
lens.add_surface(index=10)

# add aperture
lens.set_aperture(aperture_type="float_by_stop_size", value=40)

# add field
lens.set_field_type(field_type="object_height")
lens.add_field(x=0,y=0)
lens.add_field(x=10,y=0)
lens.add_field(x=20,y=0)
lens.add_field(x=-20,y=0)
#lens.add_field(y=10)

# add wavelength
lens.add_wavelength(value=0.550, is_primary=True)
lens.add_wavelength(value=0.450)
lens.add_wavelength(value=0.850)


lens.draw(
    wavelengths=[0.450, 0.550, 0.850],
    figsize=(16, 4),
    num_rays=5,
)

spot = analysis.SpotDiagram(lens,num_rings=12,wavelengths= [0.550])
spot.view()

#trace marginal ray
ray = lens.trace_generic(Hx=0, Hy=0, Px=0, Py=0, wavelength=0.587)