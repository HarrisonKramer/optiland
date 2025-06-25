# optiland/geometries/forbes/geometry.py
"""Forbes Polynomial Geometry

The Forbes geometry represents a surface defined by Forbes polynomials
superimposed on a base conic surface. The sag equation implemented follows
the formulation in the CodeV documentation and G. W. Forbes' 2011 paper
"Manufacturability estimates for optical aspheres".

z = z_base + Departure
Departure = Prefactor * ConicCorrection * PolySum(u^2)
where Prefactor = u^2(1-u^2)
"""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from .qpoly import compute_z_zprime_Q2d, Q2d_nm_c_to_a_b


class ForbesGeometry(NewtonRaphsonGeometry):
    """
    Represents a Forbes polynomial geometry.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        radius: float,
        conic: float = 0.0,
        coeffs_n=None,
        coeffs_c=None,
        norm_radius: float = 1.0,
        tol: float = 1e-10,
        max_iter: int = 100,
    ):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.coeffs_n = coeffs_n if coeffs_n is not None else []
        self.coeffs_c = be.array(coeffs_c if coeffs_c is not None else [])
        self.norm_radius = norm_radius
        self.is_symmetric = False
        self._restructure_coeffs()

    def _restructure_coeffs(self):
        """Restructures coefficients for computation."""
        if not self.coeffs_n or len(self.coeffs_c) == 0:
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = [], [], []
        else:
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = Q2d_nm_c_to_a_b(self.coeffs_n, self.coeffs_c)

    def sag(self, x=0, y=0):
        # Ensure inputs are backend-compatible arrays/tensors
        x = be.array(x)
        y = be.array(y)
        
        r2 = x**2 + y**2
        if be.isinf(self.radius):
            z_base = be.zeros_like(r2)
        else:
            sqrt_arg = 1 - (1 + self.k) * r2 / self.radius**2
            safe_sqrt_arg = be.where(sqrt_arg < 0, 0, sqrt_arg)
            z_base = r2 / (self.radius * (1 + be.sqrt(safe_sqrt_arg)))

        if not self.cm0_coeffs and not self.ams_coeffs and not self.bms_coeffs:
            return z_base

        rho = be.sqrt(r2)
        u = rho / self.norm_radius
        usq = u * u
        theta = be.arctan2(y, x)

        # compute_z_zprime_Q2d now returns the pure polynomial sum and its derivatives
        poly_sum, _, _ = compute_z_zprime_Q2d(self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs, u, theta)

        # Assemble the full departure term according to the formula:
        # Departure = Prefactor * CorrectionFactor * PolynomialSum
        prefactor = usq * (1 - usq)

        # The full departure term
        departure = be.zeros_like(rho)

        if be.isinf(self.radius):
            # Planar base, no correction factor
            departure = prefactor * poly_sum
        elif self.k == 0:
            # Spherical base, correction factor is 1/phi(rho)
            phi_sq = 1 - r2 / self.radius**2
            safe_phi = be.sqrt(be.where(phi_sq > 0, phi_sq, 1e-12))
            departure = prefactor * poly_sum / safe_phi
        else:
            # Conic base requires the full correction factor
            c2 = (1.0 / self.radius)**2
            sqrt_arg_num = 1 - c2 * self.k * r2
            sqrt_arg_den = 1 - c2 * (self.k + 1) * r2
            safe_sqrt_num = be.sqrt(be.where(sqrt_arg_num >= 0, sqrt_arg_num, 0.0))
            safe_sqrt_den = be.sqrt(be.where(sqrt_arg_den > 0, sqrt_arg_den, 1e-12))
            conic_correction_factor = safe_sqrt_num / safe_sqrt_den
            departure = prefactor * conic_correction_factor * poly_sum
        
        S = be.where(u > 1, 0.0, departure)
        return z_base + S

    def _surface_normal(self, x, y):
        """
        Calculates the cartesian components of the surface normal vector (nx, ny, nz)
        by numerically differentiating the sag function. This is more robust than an
        analytical derivative and guarantees consistency with the sag implementation.
        """
        x = be.array(x)
        y = be.array(y)
        eps = 1e-8  # A small step for numerical differentiation

        # Calculate sag at the central point
        z0 = self.sag(x, y)

        # Calculate sag at (x + eps, y) to find the partial derivative df/dx
        z_dx = self.sag(x + eps, y)
        df_dx = (z_dx - z0) / eps

        # Calculate sag at (x, y + eps) to find the partial derivative df/dy
        z_dy = self.sag(x, y + eps)
        df_dy = (z_dy - z0) / eps
        
        # The surface normal vector is proportional to (-df/dx, -df/dy, 1).
        # We then normalize this vector.
        mag = be.sqrt(df_dx**2 + df_dy**2 + 1)
        safe_mag = be.where(mag < 1e-12, 1.0, mag)

        nx_raw = -df_dx / safe_mag
        ny_raw = -df_dy / safe_mag
        nz_raw = 1.0 / safe_mag

        # --- Zemax Convention Correction ---
        # The Zemax output shows the normal vector is flipped compared to our calculation.
        # We apply a negative sign to match the expected ray-tracing convention.
        nx = -nx_raw
        ny = -ny_raw
        nz = -nz_raw

        return nx, ny, nz

    def flip(self):
        self.radius = -self.radius
        self.coordinate_system.flip()


    def __str__(self):
        return "Forbes"

    def to_dict(self):
        geometry_dict = super().to_dict()
        geometry_dict.update({
            "coeffs_n": self.coeffs_n,
            "coeffs_c": self.coeffs_c.tolist() if hasattr(self.coeffs_c, 'tolist') else self.coeffs_c,
            "norm_radius": self.norm_radius,
        })
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        cs = CoordinateSystem.from_dict(data["cs"])
        return cls(
            cs,
            data["radius"],
            data.get("conic", 0.0),
            data.get("coeffs_n", []),
            data.get("coeffs_c", []),
            data.get("norm_radius", 1.0),
            data.get("tol", 1e-10),
            data.get("max_iter", 100),
        )