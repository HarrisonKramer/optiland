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

class RadialPhase(BasePhase):
    """Represents a phase function that can be added to a Phase
    
    
    
    
    Args:
        A (float): Period of the diffraction garting in lines/um
        Order (integer): diffraction order for the grating

    Methods:
        phase_grating(self, rays): Calculates the phase for a diffraction grating
        


    """

    def __init__(self, order = 1, coef = [], eff = 'ideal'):
        
        
        self.order = be.array(order)
        self.coef = be.array(coef)
        self.eff = eff 

        
    def __str__(self):
        return "Radial"

    def phasefunction(self, x=0, y=0):
        """Calculate the surface sag of the geometry at the given coordinates.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            be.ndarray or float: The sag value(s) at the given coordinates.

        """
        m = self.order
        r = be.sqrt(x**2 + y**2)
        """Compute radial wrapped phase for order m."""
        phi_design = sum(a * r**(2*i) for i, a in enumerate(self.coef, start=1))
        phi_ordered = m * phi_design
        
        return phi_ordered

    def phase_calc(self, rays, nx, ny, nz, n1, n2):
        nx = -1*nx
        ny = -1*ny
        nz = -1*nz
        
        m = self.order
        r = be.sqrt(rays.x**2 + rays.y**2)
        """Compute radial wrapped phase for order m."""
        phi_design = sum(a * r**(2*i) for i, a in enumerate(self.coef, start=1)) 
        phi_ordered = m * (phi_design ) 

        dphi_dx = 0
        dphi_dy = 0
   
        dphi_dr = sum(2*i * a * r**(2*i - 1) for i, a in enumerate(self.coef,start=1)) 
        with be.errstate(divide='ignore', invalid='ignore'):
             dphi_dx = be.where(r != 0, dphi_dr * rays.x / r, 0.0)
             dphi_dy = be.where(r != 0, dphi_dr * rays.y / r, 0.0)

        
        # Sin = Sout + Q*N
        
        wvl = rays.w
        
        #k = 2 * be.pi / rays.w
        mu = 1.0 if wvl is None else wvl / rays.w 
        
        in_cosI = rays.L * nx + rays.M * ny + rays.N * nz
       
        
        b = in_cosI + m * (nx*dphi_dx + ny * dphi_dy)
        c = mu * (mu * (dphi_dx**2 + dphi_dy**2) / 2 + m * (rays.L * dphi_dx + rays.M * dphi_dy))
        
        discrim = b**2 - 2*c
        #if discrim < 0:
        #    raise ValeError("nogo")
        
        Q =  - b + rays.N * be.sqrt(discrim)
        
        kfx = rays.L + m * mu * dphi_dx + Q*nx
        kfy = rays.M + m * mu * dphi_dy + Q*ny
        kfz = rays.N + Q * nz
        
        out_mag = be.sqrt(kfx**2 + kfy**2 + kfz**2)
        kfx /= out_mag
        kfy /= out_mag
        kfz /= out_mag 
        
        opd =    mu * (dphi_dx + dphi_dy)
            
       
        return kfx, kfy , kfz, opd 
       

    def efficiency(self, ray):
        #need to add code to this
        if (self.eff == 'ideal'):
            d_eff = 1
            
        return d_eff
    
    def to_dict(self):
        """Convert the phase to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        phase_dict = super().to_dict()
        phase_dict.update({"period": float(self.A), "order": float(self.order)})
        return phase_dict

    @classmethod
    def from_dict(cls, data):
        """Create a phase from a dictionary.

        Args:
            data (dict): The dictionary representation of the phase.

        Returns:
            GratingPhase: An instance of GratingPhase.

        """
        required_keys = {"order", "coef"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(cs, data["coef"], data.get("order", 0.0))