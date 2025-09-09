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

    def __init__(self, A= 1, order = 1, gx = 1, gy = 0, gz = 0, eff = 'ideal'):
        
        self.A= be.array(A)
        self.order = be.array(order)
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.eff = eff 
                
    def __str__(self):
        return "Grating"

   

    def phase_calc(self, rays, nx, ny, nz, n1, n2):
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

        #self.normalize() 
        dot_knn = dx * nx + dy * ny + dz * nz
        sin_in = be.sqrt(1 - dot_knn**2)
        dot_kfn = kfx * nx + kfy * ny + kfz * nz
        sin_out = be.sqrt(1 - dot_kfn**2)
        d = 1/self.A 
        opd =  d  * (n1 * sin_in + n2 * sin_out)

        return kfx, kfy, kfz , opd 
    
    def efficiency(self, rays ):
        #need to add code to this
        if (self.eff == 'ideal'):
            d_eff = 1
            
            #beta  = d * (n1 * be.cos( rays))
            
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
        required_keys = {"order", "period","eff"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(cs, data["period"], data.get("order", 0.0), data.get("eff", 'ideal'))