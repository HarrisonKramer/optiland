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
        self.order = be.array(order)
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
        m = self.order
        r = be.sqrt(rays.x**2 + rays.y**2)
        """Compute radial wrapped phase for order m."""
        phi_design = sum(a * r**(2*i) for i, a in enumerate(self.coef, start=1))
        phi_ordered = m * phi_design
        

   
        dphi_dr = sum(2*i * a * r**(2*i - 1) for i, a in enumerate(self.coef, start=1)) * m
        with be.errstate(divide='ignore', invalid='ignore'):
            dphi_dx = be.where(r != 0, dphi_dr * rays.x / r, 0.0)
            dphi_dy = be.where(r != 0, dphi_dr * rays.y / r, 0.0)
        
        #k = 2 * be.pi / rays.w
            
        Kx = -dphi_dx 
        Ky = -dphi_dy
        Kz = be.sqrt(1 - Kx**2 - Ky**2)

        #l = rays.L + theta_x
        #m = rays.M + theta_y
        #n = rays.N + theta_z

        #uk=be.sqrt(l**2 + m**2 + n**2)

        #l = l/uk
        #m = m/uk
        #n = n/uk




        #define parameters
        dx, dy, dz = rays.L, rays.M, rays.N
        s=1
        nx, ny, nz = s*nx, s*ny, s*nz
        
        wavelength = rays.w
        # Incident wavevector (k_in = 2π/λ * direction)
        k_mag = 2 * be.pi / wavelength
        kix = k_mag * dx
        kiy = k_mag * dy
        kiz = k_mag * dz

        dot_kn = kix * nx + kiy * ny + kiz * nz
        kpx = kix - dot_kn * nx
        kpy = kiy - dot_kn * ny
        kpz = kiz - dot_kn * nz
        
        m = self.order

        kdx = kpx + m * Kx
        kdy = kpy + m * Ky
        kdz = kpz + m * Kz

        kp2 = kdx**2 + kdy**2 + kdz**2
        
        be.where(kp2 < k_mag**2)
        dk_mag2_kp2=k_mag**2 - kp2
        if be.where(dk_mag2_kp2 < 0, True, False).any():
            raise ValueError("Angular limit on Rays due to phase ")
        
        k_perp_mag =be.sqrt(dk_mag2_kp2)
       
            
            
            
        kfx =  kdx + k_perp_mag * nx
        kfy =  kdy + k_perp_mag * ny
        kfz =  kdz + k_perp_mag * nz

        uk=be.sqrt(kfx**2 + kfy**2 + kfz**2)

        kfx = kfx/uk
        kfy = kfy/uk
        kfz = kfz/uk




        opd = phi_ordered

        return kfx, kfy , kfz, opd 
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