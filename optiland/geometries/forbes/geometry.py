"""Forbes Polynomial Geometry

The Forbes geometry represents a surface defined by Forbes polynomials
superimposed on a base conic surface. The sag equation implemented follows
the formulation in the CodeV documentation and G. W. Forbes' 2011 paper
"Manufacturability estimates for optical aspheres".
"""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from .qpoly import compute_z_zprime_Q2d, Q2d_nm_c_to_a_b, compute_z_zprime_Qbfs


class ForbesQbfsGeometry(NewtonRaphsonGeometry):
    """
    Represents a Forbes polynomial geometry (rotationally symmetric Q-type).
    """
    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        radius: float,
        conic: float = 0.0,
        coeffs_n=None,  # Accepted for factory compatibility, but not used.
        coeffs_c=None,
        norm_radius: float = 1.0,
        tol: float = 1e-10,
        max_iter: int = 100,
    ):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        # For Q-BFS (m=0 only), coeffs_n is redundant. We only need coeffs_c.
        self.coeffs_c = be.array(coeffs_c if coeffs_c is not None else [])
        self.coeffs_n = coeffs_n if coeffs_n is not None else []
        self.norm_radius = norm_radius
        self.is_symmetric = True

    def sag(self, x=0, y=0):
        x = be.array(x)
        y = be.array(y)
        
        r2 = x**2 + y**2
        if be.isinf(self.radius):
            z_base = be.zeros_like(r2)
        else:
            sqrt_arg = 1 - (1 + self.k) * r2 / self.radius**2
            safe_sqrt_arg = be.where(sqrt_arg < 0, 0, sqrt_arg)
            z_base = r2 / (self.radius * (1 + be.sqrt(safe_sqrt_arg)))

        if len(self.coeffs_c) == 0 or be.all(self.coeffs_c == 0):
            return z_base

        rho = be.sqrt(r2)
        u = rho / self.norm_radius
        usq = u * u
        
        poly_sum_m0, _ = compute_z_zprime_Qbfs(self.coeffs_c, u, usq) #
        
        prefactor = usq * (1 - usq) #

        departure = be.zeros_like(rho)
        
        # Conic correction factor for the m=0 term
        if be.isinf(self.radius):
            conic_correction_factor = 1.0
        else:
            c2 = (1.0 / self.radius)**2
            sqrt_arg_num = 1 - c2 * self.k * r2
            sqrt_arg_den = 1 - c2 * (self.k + 1) * r2
            safe_sqrt_num = be.sqrt(be.where(sqrt_arg_num >= 0, sqrt_arg_num, 0.0))
            safe_sqrt_den = be.sqrt(be.where(sqrt_arg_den > 0, sqrt_arg_den, 1e-12))
            conic_correction_factor = safe_sqrt_num / safe_sqrt_den

        departure = prefactor * conic_correction_factor * poly_sum_m0
        
        S = be.where(u > 1, 0.0, departure)
        return z_base + S

    def _surface_normal(self, x, y):
        x = be.array(x)
        y = be.array(y)
        eps = 1e-8 # Using finite differences for now
        z0 = self.sag(x, y)
        z_dx = self.sag(x + eps, y)
        df_dx = (z_dx - z0) / eps
        z_dy = self.sag(x, y + eps)
        df_dy = (z_dy - z0) / eps
        mag = be.sqrt(df_dx**2 + df_dy**2 + 1)
        safe_mag = be.where(mag < 1e-12, 1.0, mag)
        nx = -(-df_dx / safe_mag)
        ny = -(-df_dy / safe_mag)
        nz = -(1.0 / safe_mag)
        return nx, ny, nz

    def flip(self):
        self.radius = -self.radius
        self.coordinate_system.flip()

    def __str__(self):
        return "ForbesQbfs"

    def to_dict(self):
        """Serializes the geometry to a dictionary."""
        geometry_dict = {
            'type': self.__class__.__name__,
            'cs': self.cs.to_dict(),
            'radius': self.radius,
            'conic': self.k,
            'tol': self.tol,
            'max_iter': self.max_iter,
            "coeffs_n": self.coeffs_n,
            "coeffs_c": self.coeffs_c.tolist() if hasattr(self.coeffs_c, 'tolist') else self.coeffs_c,
            "norm_radius": self.norm_radius,
        }
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a ForbesQbfsGeometry instance from a dictionary."""
        cs = CoordinateSystem.from_dict(data["cs"])
        return cls(
            cs,
            data["radius"],
            data.get("conic", 0.0),
            coeffs_n=data.get("coeffs_n", []),
            coeffs_c=data.get("coeffs_c", []),
            norm_radius=data.get("norm_radius", 1.0),
            tol=data.get("tol", 1e-10),
            max_iter=data.get("max_iter", 100),
        )

class ForbesQ2dGeometry(NewtonRaphsonGeometry):
    """
    Forbes Q2D aspheric surface.
    """
    def __init__(self, coordinate_system, radius, conic, coeffs_n, coeffs_c, norm_radius, tol: float = 1e-10,
        max_iter: int = 100):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.radius = float(radius)
        self.c = 1 / self.radius if self.radius != 0 else 0
        self.conic = float(conic)
        self.coeffs_n = coeffs_n
        self.coeffs_c = be.array(coeffs_c)
        self.norm_radius = float(norm_radius)

        self.cm0_coeffs = None
        self.ams_coeffs = None
        self.bms_coeffs = None

    def _prepare_coeffs(self):
        """Prepares the coefficient structure required by the qpoly module."""
        if not self.coeffs_n or len(self.coeffs_c) == 0:
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = [], [], []
        else:
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = Q2d_nm_c_to_a_b(self.coeffs_n, self.coeffs_c)

    def sag(self, x, y):
        if self.cm0_coeffs is None:
            self._prepare_coeffs()

        x = be.array(x)
        y = be.array(y)
        r2 = x**2 + y**2
        
        if be.isinf(self.radius) or self.radius == 0:
            z_base = be.zeros_like(r2)
        else:
            sqrt_arg = 1 - (1 + self.conic) * self.c**2 * r2
            safe_sqrt_arg = be.where(sqrt_arg >= 0, be.sqrt(sqrt_arg), 1.0)
            z_base = self.c * r2 / (1 + safe_sqrt_arg)
        
        rho = be.sqrt(r2)
        u = rho / self.norm_radius
        usq = u * u
        theta = be.arctan2(y, x)
        
        poly_sum_m0, _, poly_sum_m_gt0, _, _ = compute_z_zprime_Q2d(
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs, u, theta
        )
        
        # --- m=0 Departure Term ---
        departure_m0 = be.zeros_like(rho)
        has_m0_coeffs = self.cm0_coeffs is not None and len(self.cm0_coeffs) > 0 and be.any(be.array(self.cm0_coeffs) != 0)
        
        if has_m0_coeffs:
            prefactor = usq * (1 - usq)
            
            if be.isinf(self.radius):
                conic_correction_factor = 1.0
            else:
                sqrt_arg_num = 1 - self.c**2 * self.conic * r2
                sqrt_arg_den = 1 - self.c**2 * (self.conic + 1) * r2
                safe_sqrt_num = be.sqrt(be.where(sqrt_arg_num >= 0, sqrt_arg_num, 0.0))
                safe_sqrt_den = be.sqrt(be.where(sqrt_arg_den > 0, sqrt_arg_den, 1e-12))
                conic_correction_factor = safe_sqrt_num / safe_sqrt_den
            
            departure_m0 = prefactor * conic_correction_factor * poly_sum_m0

        # --- m>0 Departure Term ---
        departure_m_gt0 = poly_sum_m_gt0
        
        total_departure = departure_m0 + departure_m_gt0
        
        # The departure is only defined for u <= 1
        S = be.where(u > 1, 0.0, total_departure)
        
        return z_base + S

    def _surface_normal(self, x, y):
        x_in = be.array(x)
        y_in = be.array(y)

        # Use analytic gradients for torch backend to support autodiff.
        # Fallback to finite differences for numpy.
        if be.get_backend() == 'torch' and (x_in.requires_grad or y_in.requires_grad):
            x_local = x_in.detach().clone().requires_grad_(True)
            y_local = y_in.detach().clone().requires_grad_(True)
            
            z0 = self.sag(x_local, y_local)
            
            grad_outputs = be.ones_like(z0)
            gradients = be.autograd.grad(
                outputs=z0,
                inputs=(x_local, y_local),
                grad_outputs=grad_outputs,
                create_graph=True,
                allow_unused=True
            )
            df_dx, df_dy = gradients[0], gradients[1]
        else:
            eps = 1e-8
            z0 = self.sag(x_in, y_in)
            z_dx = self.sag(x_in + eps, y_in)
            df_dx = (z_dx - z0) / eps
            z_dy = self.sag(x_in, y_in + eps)
            df_dy = (z_dy - z0) / eps
        
        mag = be.sqrt(df_dx**2 + df_dy**2 + 1)
        safe_mag = be.where(mag < 1e-12, 1.0, mag)

        # Normal vector components
        nx = -df_dx / safe_mag
        ny = -df_dy / safe_mag
        nz = 1.0 / safe_mag

        # Optiland convention seems to be to flip the normal
        return -nx, -ny, -nz