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

from __future__ import annotations

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
    Represents a Forbes Q-bfs surface (rotationally symmetric Q-type).

    The Q-bfs surface is defined by the equation:

    z(ρ) = z_base(ρ) + (1/σ(ρ)) * [u²(1-u²) * Σ a_m Q_m(u²)]

    where:
    - z_base(ρ) = c*ρ²/(1 + √(1 - (1+k)c²ρ²)) is the base conic
    - c = 1/R is the curvature (R is the radius)
    - k is the conic constant
    - u = ρ/ρ_max is the normalized radial coordinate
    - ρ_max is the normalization radius
    - Q_m(u²) are the Forbes orthogonal polynomials
    - a_m are the polynomial coefficients
    - σ(ρ) = √(1 - c²ρ²) is a scaling factor

    Parameters
    ----------
    coordinate_system : CoordinateSystem
        The coordinate system for the surface.
    radius : float
        The vertex radius of curvature (R = 1/c).
    conic : float, optional
        The conic constant (k), by default 0.0.
    radial_terms : dict, optional
        A dictionary mapping the radial order `n` to the coefficient `a_n`.
        Example: `{0: 0.01, 2: -0.005}` for coefficients a₀ and a₂.
        Defaults to None.
    norm_radius : float, optional
        The normalization radius (ρ_max) used to define u = ρ/ρ_max, by default 1.0.
    tol : float, optional
        Tolerance for Newton-Raphson iteration, by default 1e-10.
    max_iter : int, optional
        Maximum number of iterations for Newton-Raphson, by default 100.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        radius: float,
        conic: float = 0.0,
        radial_terms: dict[int, float] | None = None,
        norm_radius: float = 1.0,
        tol: float = 1e-10,
        max_iter: int = 100,
    ):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.radial_terms = radial_terms if radial_terms else {}
        self._prepare_coeffs()
        self.norm_radius = be.array(norm_radius)
        self.is_symmetric = True

    def _prepare_coeffs(self):
        """
        Prepares the internal coefficient lists (coeffs_c, coeffs_n) from the
        user-facing radial_terms dictionary. This ensures that sparse coefficient
        dictionaries are correctly converted to the dense list format required
        by the backend polynomial evaluation functions.
        """
        if self.radial_terms:
            max_n = max(self.radial_terms.keys()) if self.radial_terms else -1
            if max_n >= 0:
                coeffs_c = [self.radial_terms.get(n, 0.0) for n in range(max_n + 1)]
                coeffs_n = [(n, 0) for n in range(max_n + 1)]
                self.coeffs_n = coeffs_n
                self.coeffs_c = be.array(coeffs_c)
            else:
                self.coeffs_n = []
                self.coeffs_c = be.array([])
        else:
            self.coeffs_n = []
            self.coeffs_c = be.array([])

    def sag(self, x=0, y=0):
        """
        Calculate the sag of the Forbes Q-bfs surface at given coordinates.

        For a Q-bfs surface, this implements:
        z(ρ) = z_base(ρ) + departure(ρ)

        where departure is defined as:
        departure(ρ) = (1/σ(ρ)) * [u²(1-u²) * Σ a_m Q_m(u²)]

        Parameters
        ----------
        x : float or array_like, optional
            X coordinate(s), by default 0
        y : float or array_like, optional
            Y coordinate(s), by default 0

        Returns
        -------
        float or array_like
            The sag value(s) at the specified coordinates
        """
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

        The normal vector is based on the gradient of the sag function:
        n = [-dz/dx, -dz/dy, 1]/√(1 + (dz/dx)² + (dz/dy)²)

        Parameters
        ----------
        x : float or array_like
            X coordinate(s)
        y : float or array_like
            Y coordinate(s)

        Returns
        -------
        tuple
            (nx, ny, nz) components of the unit normal vector
        """
        x_in = be.array(x)
        y_in = be.array(y)

        # If the backend is torch, use its autograd engine on the sag function
        if be.get_backend() == "torch":
            x_grad = x_in.clone().requires_grad_(True)
            y_grad = y_in.clone().requires_grad_(True)

            z0 = self.sag(x_grad, y_grad)

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

        else:
            r2 = x_in**2 + y_in**2
            rho = be.sqrt(r2)

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

                # prefactor = u^2 - u^4
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

    def __str__(self):
        return "ForbesQbfs"

    def to_dict(self):
        """
        Serializes the geometry to a dictionary.

        Returns
        -------
        dict
            Dictionary containing all parameters needed to reconstruct the geometry
        """
        geometry_dict = {
            "type": self.__class__.__name__,
            "cs": self.cs.to_dict(),
            "radius": self.radius,
            "conic": self.k,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "radial_terms": self.radial_terms,
            "norm_radius": self.norm_radius,
        }
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """
        Creates a ForbesQbfsGeometry instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing the geometry parameters

        Returns
        -------
        ForbesQbfsGeometry
            The reconstructed geometry instance
        """
        cs = CoordinateSystem.from_dict(data["cs"])
        return cls(
            cs,
            data["radius"],
            data.get("conic", 0.0),
            radial_terms=data.get("radial_terms", None),
            norm_radius=data.get("norm_radius", 1.0),
            tol=data.get("tol", 1e-10),
            max_iter=data.get("max_iter", 100),
        )


class ForbesQ2dGeometry(NewtonRaphsonGeometry):
    """
    Forbes Q2D freeform surface.

    This represents the most general form of a Forbes surface that can describe
    a freeform optic without rotational symmetry. The sag is defined as:

    z(ρ,θ) = z_base(ρ) + (1/√(1-c²ρ²)) * δ(u,θ)

    where δ(u,θ) is the departure function:

    δ(u,θ) = u²(1-u²)∑_{n=0}^N a_n^0 Q_n^0(u²) +
             ∑_{m=1}^M u^m ∑_{n=0}^N [a_n^m cos(mθ) + b_n^m sin(mθ)] Q_n^m(u²)

    The first term represents the rotationally symmetric part, and the second
    term represents the non-symmetric (freeform) part.

    Parameters
    ----------
    coordinate_system : CoordinateSystem
        The coordinate system for the surface.
    radius : float
        The vertex radius of curvature (R = 1/c).
    conic : float
        The conic constant (k).
    freeform_coeffs : dict, optional
        A dictionary defining the freeform surface coefficients.
        Keys are tuples:
        - `(n, m)` for a cosine term (a_n^m).
        - `(n, m, 'sin')` for a sine term (b_n^m).
        Values are the float coefficient values.
        Example: `{(2, 2): 0.05, (3, 1, 'sin'): 0.02}`.
        Defaults to None.
    norm_radius : float, optional
        The normalization radius (ρ_max) used to define u = ρ/ρ_max, by default 1.0.
    tol : float, optional
        Tolerance for Newton-Raphson iteration, by default 1e-10.
    max_iter : int, optional
        Maximum number of iterations for Newton-Raphson, by default 100.
    """

    def __init__(
        self,
        coordinate_system,
        radius,
        conic,
        freeform_coeffs=None,
        norm_radius=1.0,
        tol: float = 1e-10,
        max_iter: int = 100,
    ):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.radius = float(radius)
        self.c = 1 / self.radius if self.radius != 0 else 0
        self.conic = float(conic)
        self.freeform_coeffs = freeform_coeffs if freeform_coeffs else {}
        self.norm_radius = float(norm_radius)

        # Initialize internal lists that will be populated by _prepare_coeffs
        self.coeffs_n = []
        self.coeffs_c = be.array([])
        self.cm0_coeffs = []
        self.ams_coeffs = []
        self.bms_coeffs = []

        # Call prepare_coeffs to populate everything from freeform_coeffs
        self._prepare_coeffs()

    def _prepare_coeffs(self):
        """
        Prepares all internal coefficient structures (coeffs_n, coeffs_c, and
        the qpoly backend lists) from the user-facing freeform_coeffs dictionary.
        This method is the single source of truth for converting the user-friendly
        dictionary into all necessary internal formats.
        """
        if self.freeform_coeffs:
            coeffs_n = []
            coeffs_c = []
            # Sort by n, then m, then by type (cos/sin) to ensure consistent ordering
            sorted_keys = sorted(
                self.freeform_coeffs.keys(),
                key=lambda k: (k[0], abs(k[1]), 0 if len(k) == 2 else 1),
            )
            for key in sorted_keys:
                value = self.freeform_coeffs[key]
                if len(key) == 3 and key[2].lower() == "sin":
                    coeffs_n.append((key[0], -key[1]))
                else:
                    coeffs_n.append((key[0], key[1]))
                coeffs_c.append(value)

            self.coeffs_n = coeffs_n
            self.coeffs_c = be.array(coeffs_c)
        else:
            self.coeffs_n = []
            self.coeffs_c = be.array([])

        # Now, prepare the backend-specific lists required by qpoly
        if self.coeffs_n and len(self.coeffs_c) > 0:
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = Q2d_nm_c_to_a_b(
                self.coeffs_n, self.coeffs_c
            )
        else:
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = [], [], []

    def sag(self, x, y):
        """
        Calculate the sag of the Forbes Q2D freeform surface at given coordinates.

        This implements the full freeform surface equation:
        z(ρ,θ) = z_base(ρ) + (1/√(1-c²ρ²)) * δ(u,θ)

        with both rotationally symmetric and non-symmetric components.

        Parameters
        ----------
        x : float or array_like
            X coordinate(s)
        y : float or array_like
            Y coordinate(s)

        Returns
        -------
        float or array_like
            The sag value(s) at the specified coordinates
        """
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

    def to_dict(self):
        """
        Serializes the geometry to a dictionary.

        Returns
        -------
        dict
            Dictionary containing all parameters needed to reconstruct the geometry
        """
        return {
            "type": self.__class__.__name__,
            "cs": self.cs.to_dict(),
            "radius": self.radius,
            "conic": self.conic,
            "freeform_coeffs": self.freeform_coeffs,
            "norm_radius": self.norm_radius,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Creates a ForbesQ2dGeometry instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing the geometry parameters

        Returns
        -------
        ForbesQ2dGeometry
            The reconstructed geometry instance
        """
        cs = CoordinateSystem.from_dict(data["cs"])
        return cls(
            cs,
            data["radius"],
            data.get("conic", 0.0),
            freeform_coeffs=data.get("freeform_coeffs", None),
            norm_radius=data.get("norm_radius", 1.0),
            tol=data.get("tol", 1e-10),
            max_iter=data.get("max_iter", 100),
        )

    def __str__(self):
        return "ForbesQ2d"
