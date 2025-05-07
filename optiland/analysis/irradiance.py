"""Irradiance Analysis

This module implements the necessary logic for the 
irradiance analysis in a given optical system. 
*note*: for now we only take of the incoherent irradiance. 

The analysis is analogous to the SpotDiagram except that 
instead of plotting the landing position of individual rays, 
we accumulate their power on a detector and express the result 
in W/mm^-2.


Manuel Fragata Mendes, 2025
"""

import matplotlib.pyplot as plt
from typing import List, Tuple

import numpy as _np # Use _np for the binning. Later extend to 
                    # other backends
import optiland.backend as be 

class IncoherentIrradiance:
    """Compute and visualise incoherent irradiance on the detector surface.
       For simplification, we assume that the detector surface = image surface.

        Attributes: 
        --- 
        optic : optiland.optic.Optic
            Reference to the optical system - must already define fields, wavelengths
            and, critically, a physical aperture on the chosen detector surface.
        res : tuple[int, int]
            Requested pixel count along (x,y) of the irradiance grid.
        px_size : tuple[float, float] | None
            Physical pixel pitch (dx,dy) in mm.  If ``None`` the pitch is
            derived from the surface aperture and `res`.
        n_rays : int
            Number of real rays launched for every (field,wavelength) pair.
        fields, wavelengths : tuple | "all"
            Convenience selectors that work exactly like those in
            `SpotDiagram` - default is to analyse all of them.
        detector_surface : int
            Index into `optic.surface_group.surfaces` that designates the detector
            plane to analyse (default=`-1`->image surface).
        irr_data : list[list[be.ndarray]]
            2-D irradiance arrays for every (field,wvl) - outer index is field,
            inner index is wavelength.  Each array has shape
            (res[0],res[1]) with X as the row index so that
            ``irr_data[f][w][i,j]`` refers to X=i, Y=j.
        
        Methods
        ---
        view(figsize=(6,5), cmap="inferno") → None
            Display false-colour irradiance maps three fields per row, sharing a common
            colour bar.
        peak_irradiance() → list[list[float]]
            Return the maximum pixel value for every (field,wvl) pair.
    """
    
    def __init__(self,optic, 
                 n_rays: int, 
                 res = (128,128),
                 px_size : float = None,
                 detector_surface : int = -1,
                 *,
                 fields = "all",
                 wavelengths = "all",
                 distribution : str = "random",
                 ):
        
        self.optic = optic
        self.n_rays = n_rays
        self.npix_x, self.npix_y = res
        self.px_size = None if px_size is None else (float(px_size[0]), float(px_size[1]))
        self.detector_surface = int(detector_surface)

        self.fields = (optic.fields.get_field_coords() if fields == "all" else tuple(fields))
        self.wavelengths = (optic.wavelengths.get_wavelengths() if wavelengths == "all" else tuple(wavelengths))

        # the detector surface must have a physical aperture 
        surf = self.optic.surface_group.surfaces[self.detector_surface]
        if surf.aperture is None:
            raise ValueError("Detector surface has no physical aperture – set one "
                "(e.g. RectangularAperture) so that the detector size is defined.")

        # Generate irradiance for every (field, wvl) pair
        self.irr_data = (self._generate_data(distribution))

    # ADD: utility functions, view(), etc

    # helper functions

    def peak_irradiance(self):
        """Maximum pixel value for each (field,wvl) pair."""
        return [[be.max(irr) for irr, *_ in fblock] for fblock in self.irr_data]

    # data generation functions

    def _generate_data(self, distribution):
        data = []
        for field in self.fields:
            f_block = []
            for wl in self.wavelengths:
                f_block.append(self._single_field_wl(field, wl, distribution))
            data.append(f_block)
        return data

    def _single_field_wl(self, field, wavelength, distribution):
        """Trace rays and bin their power into the pixels of the detector."""
        Hx, Hy = field
        self.optic.trace(Hx, Hy, wavelength, self.n_rays, distribution)

        # get ray coords on detector surface 
        surf = self.optic.surface_group.surfaces[self.detector_surface]
        x_g, y_g, z_g = surf.x, surf.y, surf.z
        power = surf.intensity

        from optiland.visualization.utils import transform 

        x_local, y_local, _ = transform(x_g, y_g, z_g, surf, is_global=True)
        x_np = be.to_numpy(x_local)
        y_np = be.to_numpy(y_local)
        power_np = be.to_numpy(power)

        valid = power_np > 0.0
        x_np, y_np, power_np = x_np[valid], y_np[valid], power_np[valid]

        # get the physical siize of the detector
        x_min, x_max, y_min, y_max = surf.aperture.extent
        if self.px_size is None:
            x_edges = _np.linspace(x_min, x_max, self.npix_x + 1, dtype=float)
            y_edges = _np.linspace(y_min, y_max, self.npix_y + 1, dtype=float)
            pixel_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        else:
            dx, dy = self.px_size
            x_edges = _np.arange(x_min, x_max + 0.5 * dx, dx, dtype=float)
            y_edges = _np.arange(y_min, y_max + 0.5 * dy, dy, dtype=float)
            pixel_area = dx * dy
            # if the pitch supplied by the user gives a different res 
            # than the one requested warn once
            exp_nx = len(x_edges) - 1
            exp_ny = len(y_edges) - 1
            if (exp_nx, exp_ny) != (self.npix_x, self.npix_y):
                print("[IncoherentIrradiance] Warning: res parameter ignored - "
                    "derived from px_size instead → (%d,%d) pixels" % (exp_nx, exp_ny))
                self.npix_x, self.npix_y = exp_nx, exp_ny

        # 2d binning with numpy histogram
        hist, _, _ = _np.histogram2d(x_np, y_np, bins=[x_edges, y_edges], weights=power_np)
        irr = hist / pixel_area
        return be.array(irr), x_edges, y_edges
    
    
    
    

