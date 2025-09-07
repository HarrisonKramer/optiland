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
        self.coef = coef
 



        
    def __str__(self):
        return "Radial"

   

    def phase_calc(self, rays, coeffs, m):
        r = be.sqrt(rays.x**2 + rays.y**2)
        """Compute radial wrapped phase for order m."""
        phi_design = sum(a * r**(2*i) for i, a in enumerate(coeffs, start=1))
        phi_ordered = m * phi_design
        

   
        dphi_dr = sum(2*i * a * r**(2*i - 1) for i, a in enumerate(coeffs, start=1)) * m
        with np.errstate(divide='ignore', invalid='ignore'):
            dphi_dx = np.where(r != 0, dphi_dr * x / r, 0.0)
            dphi_dy = np.where(r != 0, dphi_dr * y / r, 0.0)
        
        k = 2 * be.pi / rays.w
            
        theta_x = -dphi_dx / k
        theta_y = -dphi_dy / k
        theta_z = be.sqrt(1 - theta_x**2 - theta_y**2)

        l = rays.L + theta_x
        m = rays.M + theta_y
        n = rays.N + theta_z

        uk=be.sqrt(l**2 + m**2 + n**2)

        l = l/uk
        m = m/uk
        n = n/uk

        opd = phi_ordered

        return l, m , n, opd 
        #return phi_design

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