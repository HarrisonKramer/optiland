"""Forbes Polynomial Geometries for Optical Surfaces

This module provides implementations for optical surfaces described by
Forbes polynomials, which are superimposed on a base conic section.
Forbes polynomials offer a modern alternative to standard power-series
aspheres, providing better control over the surface shape and its derivatives.
Their orthogonality helps to decouple the effects of different coefficients,
which is advantageous for optimization, tolerancing, and assessing
manufacturability.

The implementation is based on the work of G. W. Forbes. See, for example,
G. W. Forbes, "Manufacturability estimates for optical aspheres," Opt. Express (2011).

This module implements two types of Forbes surfaces:

1.  **ForbesQbfsGeometry (Q-type, Best Fit Sphere)**:
    This class represents rotationally symmetric aspheric surfaces. The "Qbfs"
    polynomials are orthogonal over a circular aperture and describe the
    departure from a base conic.

2.  **ForbesQ2dGeometry (Q-type, 2D)**:
    This class represents non-rotationally symmetric, or "freeform," surfaces.
    The Q2D polynomials extend the Q-type formalism to two dimensions, allowing
    for the description of complex, freeform optical surfaces that lack
    rotational symmetry.

Manuel Fragata Mendes, 2025
"""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry

from .qpoly import (
    Q2d_nm_c_to_a_b,
    clenshaw_qbfs,
    compute_z_zprime_Q2d,
    compute_z_zprime_Qbfs,
)


class ForbesQbfsGeometry(NewtonRaphsonGeometry):
    """
    Represents a Forbes polynomial geometry (rotationally symmetric Q-type).
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
        # For Q-BFS (m=0 only), coeffs_n is redundant. We only need coeffs_c.
        self.coeffs_c = be.array(coeffs_c if coeffs_c is not None else [])
        self.coeffs_n = coeffs_n if coeffs_n is not None else []
        self.norm_radius = be.array(norm_radius)
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

        poly_sum_m0 = clenshaw_qbfs(self.coeffs_c, usq)

        prefactor = usq * (1 - usq)

        # Conic correction factor for the m=0 term
        if be.isinf(self.radius):
            conic_correction_factor = 1.0
        else:
            c2 = (1.0 / self.radius) ** 2
            sqrt_arg_num = 1 - c2 * self.k * r2
            sqrt_arg_den = 1 - c2 * (self.k + 1) * r2
            safe_sqrt_num = be.sqrt(be.where(sqrt_arg_num >= 0, sqrt_arg_num, 0.0))
            safe_sqrt_den = be.sqrt(be.where(sqrt_arg_den > 0, sqrt_arg_den, 1e-12))
            conic_correction_factor = safe_sqrt_num / safe_sqrt_den

        departure = prefactor * conic_correction_factor * poly_sum_m0

        S = be.where(usq > 1, 0.0, departure)
        return z_base + S

    def _surface_normal(self, x, y):
        """
        Calculates the surface normal. Uses automatic differentiation for PyTorch
        and the analytical derivative for NumPy to ensure correctness and performance.
        """
        x_in = be.array(x)
        y_in = be.array(y)

        # If the backend is torch, use its autograd engine on the sag function
        if be.get_backend() == "torch":
            x_grad = x_in.clone().requires_grad_(True)
            y_grad = y_in.clone().requires_grad_(True)

            # --- Forward Pass ---
            # Calculate sag for all points simultaneously.
            z0 = self.sag(x_grad, y_grad)

            # --- Backward Pass ---
            # Compute gradients for all points. This will produce NaNs at the vertex.
            gradients = be.autograd.grad(
                outputs=z0,
                inputs=(x_grad, y_grad),
                grad_outputs=be.ones_like(z0),
                create_graph=True,
                allow_unused=True,
            )
            df_dx_raw, df_dy_raw = gradients[0], gradients[1]

            # identify vertex points where x and y are both 0
            is_vertex = be.logical_and(x_in == 0.0, y_in == 0.0)

            # correct the derivatives at the vertex to avoid NaNs
            df_dx = be.where(is_vertex, 0.0, df_dx_raw)
            df_dy = be.where(is_vertex, 0.0, df_dy_raw)
        # <<< KEEP ORIGINAL CODE IN ELSE BLOCK FOR NUMPY >>>
        # use the original analytical derivative if not using torch
        else:
            r2 = x_in**2 + y_in**2
            rho = be.sqrt(r2)

            # Avoid division by zero at the vertex
            rho_safe = be.where(rho < 1e-9, 1e-9, rho)

            # derivative of the base conic sag (dS_base / d(rho))
            if be.isinf(self.radius) or self.radius == 0:
                dS_base_d_rho = be.zeros_like(rho)
            else:
                c = 1.0 / self.radius
                sqrt_arg_base = 1 - (self.k + 1) * c**2 * r2
                safe_sqrt_base = be.sqrt(
                    be.where(sqrt_arg_base > 0, sqrt_arg_base, 1e-12)
                )
                dS_base_d_rho = c * rho / safe_sqrt_base

            # if no coefficients, derivative is just from the base conic
            if len(self.coeffs_c) == 0 or be.all(self.coeffs_c == 0):
                df_d_rho = dS_base_d_rho
            else:
                # derivative of the departure term
                a = self.norm_radius
                u = rho / a
                usq = u * u

                # Get polynomial value and its derivative w.r.t. u from qpoly
                poly_val, dPoly_d_u = compute_z_zprime_Qbfs(self.coeffs_c, u, usq)

                # Prefactor = u^2 - u^4
                dPrefactor_d_rho = (2 * u - 4 * u**3) / a

                # d(Poly) / d(rho)
                dPoly_d_rho = dPoly_d_u / a

                if be.isinf(self.radius):
                    conic_factor = 1.0
                    dConicFactor_d_rho = 0.0
                else:
                    c2 = (1.0 / self.radius) ** 2
                    num_arg = 1 - self.k * c2 * r2
                    den_arg = 1 - (self.k + 1) * c2 * r2

                    safe_num_arg = be.where(num_arg > 0, num_arg, 1e-12)
                    safe_den_arg = be.where(den_arg > 0, den_arg, 1e-12)

                    N = be.sqrt(safe_num_arg)
                    D = be.sqrt(safe_den_arg)

                    conic_factor = N / D
                    # simplified derivative of the conic factor w.r.t rho
                    dConicFactor_d_rho = (c2 * rho) / (N * D**3)

                # full derivative of departure using product rule
                # for (Prefactor * ConicFactor * Poly)
                dS_dep_d_rho = (
                    dPrefactor_d_rho * conic_factor * poly_val
                    + (u**2 - u**4) * dConicFactor_d_rho * poly_val
                    + (u**2 - u**4) * conic_factor * dPoly_d_rho
                )

                # Total derivative - apply departure derivative
                # only within normalization radius
                df_d_rho = dS_base_d_rho + be.where(u >= 1, 0.0, dS_dep_d_rho)

            # Convert derivative w.r.t. rho to derivatives w.r.t. x and y
            df_dx = df_d_rho * (x_in / rho_safe)
            df_dy = df_d_rho * (y_in / rho_safe)

        # Normalize to get the final normal vector components
        mag = be.sqrt(df_dx**2 + df_dy**2 + 1)
        safe_mag = be.where(mag < 1e-12, 1.0, mag)

        nx = df_dx / safe_mag
        ny = df_dy / safe_mag
        nz = -1 / safe_mag
        return nx, ny, nz

    def flip(self):
        self.radius = -self.radius
        self.coordinate_system.flip()

    def __str__(self):
        return "ForbesQbfs"

    def to_dict(self):
        """Serializes the geometry to a dictionary."""
        geometry_dict = {
            "type": self.__class__.__name__,
            "cs": self.cs.to_dict(),
            "radius": self.radius,
            "conic": self.k,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "coeffs_n": self.coeffs_n,
            "coeffs_c": self.coeffs_c.tolist()
            if hasattr(self.coeffs_c, "tolist")
            else self.coeffs_c,
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

    def __init__(
        self,
        coordinate_system,
        radius,
        conic,
        coeffs_n,
        coeffs_c,
        norm_radius,
        tol: float = 1e-10,
        max_iter: int = 100,
    ):
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
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = Q2d_nm_c_to_a_b(
                self.coeffs_n, self.coeffs_c
            )

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
        safe_x = be.where((rho < 1e-9), x + 1e-12, x)
        theta = be.arctan2(y, safe_x)

        poly_sum_m0, _, poly_sum_m_gt0, _, _ = compute_z_zprime_Q2d(
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs, u, theta
        )

        # --- m=0 Departure Term ---
        departure_m0 = be.zeros_like(rho)
        has_m0_coeffs = (
            self.cm0_coeffs is not None
            and len(self.cm0_coeffs) > 0
            and be.any(be.array(self.cm0_coeffs) != 0)
        )

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

        # if the backend is torch, use its autodiff for exact gradients
        if be.get_backend() == "torch" and (x_in.requires_grad or y_in.requires_grad):
            z0 = self.sag(x_in, y_in)

            # compute d(z0)/dx and d(z0)/dy
            gradients = be.autograd.grad(
                outputs=z0,
                inputs=(x_in, y_in),
                grad_outputs=be.ones_like(z0),
                create_graph=True,
                allow_unused=True,
            )
            df_dx, df_dy = gradients[0], gradients[1]

        else:
            if self.cm0_coeffs is None:
                self._prepare_coeffs()

            r2 = x_in**2 + y_in**2
            rho = be.sqrt(r2)
            rho_safe = be.where(rho < 1e-9, 1e-9, rho)

            u = rho / self.norm_radius
            safe_x = be.where(rho < 1e-9, 1e-9, x_in)
            theta = be.arctan2(y_in, safe_x)

            # Get derivatives from qpoly
            _, d_poly_m0_du, _, dr_poly_m_gt0, dt_poly_m_gt0 = compute_z_zprime_Q2d(
                self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs, u, theta
            )

            # m=0 part (from ForbesQbfs logic)
            dS0_d_rho = be.zeros_like(rho)
            if self.cm0_coeffs is not None and len(self.cm0_coeffs) > 0:
                poly_val_m0, _ = compute_z_zprime_Qbfs(self.cm0_coeffs, u, u**2)
                # identical to the forbesQbfs logic
                a = self.norm_radius
                dPrefactor_d_rho = (2 * u - 4 * u**3) / a
                dPoly_d_rho_m0 = d_poly_m0_du / a
                if be.isinf(self.radius):
                    conic_factor = 1.0
                    dConic_d_rho = 0.0
                else:
                    c2 = self.c**2
                    k = self.conic
                    n_arg = 1 - k * c2 * r2
                    d_arg = 1 - (k + 1) * c2 * r2
                    N = be.sqrt(be.where(n_arg > 0, n_arg, 1e-12))
                    D = be.sqrt(be.where(d_arg > 0, d_arg, 1e-12))
                    conic_factor = N / D
                    dConic_d_rho = (c2 * rho) / (N * D**3)

                dS0_d_rho = (
                    dPrefactor_d_rho * conic_factor * poly_val_m0
                    + (u**2 - u**4) * dConic_d_rho * poly_val_m0
                    + (u**2 - u**4) * conic_factor * dPoly_d_rho_m0
                )

            # m>0 part
            dS_d_rho = be.where(u > 1, 0, dS0_d_rho + dr_poly_m_gt0)
            dS_d_theta = be.where(u > 1, 0, dt_poly_m_gt0)

            #  full derivative using chain rule for polar coords
            # d/dx = cos(t)*d/d(rho) - sin(t)/rho*d/d(theta)
            # d/dy = sin(t)*d/d(rho) + cos(t)/rho*d/d(theta)
            cos_t = x_in / rho_safe
            sin_t = y_in / rho_safe

            dS_dx = cos_t * dS_d_rho - (sin_t / rho_safe) * dS_d_theta
            dS_dy = sin_t * dS_d_rho + (cos_t / rho_safe) * dS_d_theta

            # Base conic derivative
            if be.isinf(self.radius) or self.radius == 0:
                d_base_dx = be.zeros_like(x_in)
                d_base_dy = be.zeros_like(y_in)
            else:
                sqrt_arg_base = 1 - (1 + self.conic) * self.c**2 * r2
                safe_sqrt_base = be.sqrt(
                    be.where(sqrt_arg_base > 0, sqrt_arg_base, 1e-12)
                )
                d_base_dr = (self.c * rho) / safe_sqrt_base
                d_base_dx = d_base_dr * cos_t
                d_base_dy = d_base_dr * sin_t

            df_dx = d_base_dx + dS_dx
            df_dy = d_base_dy + dS_dy

        mag = be.sqrt(df_dx**2 + df_dy**2 + 1)
        safe_mag = be.where(mag < 1e-12, 1.0, mag)

        nx = df_dx / safe_mag
        ny = df_dy / safe_mag
        nz = -1.0 / safe_mag

        return nx, ny, nz
