"""Phase 

TPhaser that gets added to any surface defiend in Geometry:
Write some kind of disccription


Hhsoj, 2025
"""

import warnings
import math as mth
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.phase.base import BasePhase

class GratingPhase(BasePhase):
    """Represents a phase function that can be added to a Geometry
    
    
    
    
    Args:
        A (float): Period of the diffraction garting in lines/um
        Order (integer): diffraction order for the grating

    Methods:
        phase_grating(self, rays): Calculates the phase for a diffraction grating
        


    """

    def __init__(self, A= 1, order = 1, gx = 1, gy = 0, gz = 0):
        
        self.A= be.array(A)
        self.order = be.array(order)
        self.gx = gx
        self.gy = gy
        self.gz = gz
        
    def __str__(self):
        return "Grating"

   

    def phase_grating_general(self, rays, nx, ny, nz):
        """"Phase function that discribes a diffraction grating
        Args:"""
        spacing = 1/self.A
        nx = -nx
        ny = -ny
        nz = -nz
        #cross product of n x g
        tx = ny * self.gz - nz * self.gy
        ty = nz * self.gx - nx * self.gz
        tz = nx * self.gy - ny * self.gx
        #tx, ty, tz = normalize3(tx, ty, tz)
        mag = be.sqrt(tx*tx + ty*ty + tz*tz)
        
 
        tx = be.where(mag <= 0, 0, tx/ mag)
        ty = be.where(mag <= 0, 0, ty/ mag)
        tz = be.where(mag <= 0, 0, tz/ mag)
        
        
        Kx = (2 * be.pi / spacing) * tx
        Ky = (2 * be.pi / spacing) * ty
        Kz = (2 * be.pi / spacing) * tz
        return Kx, Ky, Kz 

   