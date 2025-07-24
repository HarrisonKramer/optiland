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
lens.add_surface(index=3, radius=100, thickness=30, surface_type="grating", grating_order = 1, grating_period = 3.0, is_stop=True)
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

#trace marginal ray
ray = lens.trace_generic(Hx=0, Hy=0, Px=0, Py=0, wavelength=0.587)