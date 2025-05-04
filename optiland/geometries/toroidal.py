"""Toroidal Geometry

This module defines a toroidal surface geometry based on the Zemax definition:
A base curve in the YZ plane (conic + polynomials) is rotated around an
axis parallel to Y, offset by the radius of rotation R along Z.

z_y = yz_sag(y) = (c*y^2)/(1 + sqrt(1-(1+k)*c^2*y^2)) + sum(alpha_i * y^(2i+2))
z(x,y) = R - sqrt((R - z_y)^2 - x^2)

where:
- R is the radius of rotation (X-Z curvature radius at vertex)
- R_y = 1/c is the Y-Z curvature radius at vertex
- k is the conic constant for the YZ curve
- alpha_i are polynomial coefficients for the YZ curve

Manuel Fragata Mendes, 2025
"""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class ToroidalGeometry(NewtonRaphsonGeometry):
    """
    Represents a simplified toroidal geometry (no Zernike terms as in zemax -
    - may be added later if necessary).

    Args:
        coordinate_system (CoordinateSystem): The coordinate system.
        radius_rotation (float): Radius of rotation R (X-Z radius).
        radius_yz (float): Base Y-Z radius R_y.
        conic (float, optional): Conic constant k for the Y-Z curve. Defaults to 0.0.
        coeffs_poly_y (list[float], optional): Polynomial coefficients alpha_i
            for the Y-Z curve (powers y^2, y^4, ...). Defaults to [].
        tol (float, optional): Newton-Raphson tolerance. Defaults to 1e-10.
        max_iter (int, optional): Newton-Raphson max iterations. Defaults to 100.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        radius_rotation: float,
        radius_yz: float,
        conic: float = 0.0,
        coeffs_poly_y: list[float] = None,
        tol: float = 1e-10,
        max_iter: int = 100,
    ):
        # Pass radius_rotation as the base 'radius' for NewtonRaphsonGeometry.
        super().__init__(
            coordinate_system, radius_rotation, 0.0, tol, max_iter
        )  # Pass 0 for base conic

        self.R_rot = radius_rotation
        self.R_yz = radius_yz
        self.k_yz = conic

        self.coeffs_poly_y = be.asarray([] if coeffs_poly_y is None else coeffs_poly_y)

        self.is_symmetric = False

        self.c_yz = (
            1.0 / self.R_yz if be.isfinite(self.R_yz) and self.R_yz != 0 else 0.0
        )
        self.eps = 1e-14  # safe div

    def _calculate_zy(self, y: be.ndarray) -> be.ndarray:
        """Calculates the sag of the base Y-Z curve."""
        y2 = y**2
        z_y = be.zeros_like(y)

        # Base YZ conic sag part
        if be.isfinite(self.R_yz) and self.R_yz != 0:
            c = self.c_yz
            k = self.k_yz
            root_term_val = 1.0 - (1.0 + k) * c**2 * y2
            # Ensure root term is non-negative
            root_term = be.where(root_term_val < 0, 0.0, root_term_val)
            sqrt_val = be.sqrt(root_term)
            denom = 1.0 + sqrt_val
            # Avoid division by zero
            safe_denom = be.where(be.abs(denom) < self.eps, self.eps, denom)
            z_y = (c * y2) / safe_denom

        # Add YZ polynomial terms (alpha_i for y^(2i), i >= 1)
        # coeffs_poly_y[i] is coeff for y^(2*(i+1))
        if len(self.coeffs_poly_y) > 0:
            poly_term = be.zeros_like(y)
            current_y_power = y2  # Start with y^2
            for coeff in self.coeffs_poly_y:
                poly_term = poly_term + coeff * current_y_power
                current_y_power = (
                    current_y_power * y2
                )  # Increase power by y^2 for next term
            z_y = z_y + poly_term

        return z_y

    def _calculate_zy_derivative(self, y: be.ndarray) -> be.ndarray:
        """Calculates the derivative dz_y/dy of the base Y-Z curve."""
        y2 = y**2
        dz_dy = be.zeros_like(y)

        # Derivative of base YZ conic sag part
        if be.isfinite(self.R_yz) and self.R_yz != 0:
            c = self.c_yz
            k = self.k_yz
            root_term_val = 1.0 - (1.0 + k) * c**2 * y2
            # non-negative for sqrt and non-zero for division
            root_term = be.where(root_term_val < self.eps, self.eps, root_term_val)
            sqrt_val = be.sqrt(root_term)
            safe_sqrt_val = be.where(be.abs(sqrt_val) < self.eps, self.eps, sqrt_val)
            dz_dy = (c * y) / safe_sqrt_val

        # Derivative of YZ polynomial terms (alpha_i for y^(2(i+1)))
        if len(self.coeffs_poly_y) > 0:
            poly_deriv_term = be.zeros_like(y)
            current_y_power_deriv = y
            for i, coeff in enumerate(self.coeffs_poly_y):
                power_coeff = 2.0 * (i + 1.0)
                poly_deriv_term = (
                    poly_deriv_term + coeff * power_coeff * current_y_power_deriv
                )
                current_y_power_deriv = current_y_power_deriv * y2
            dz_dy = dz_dy + poly_deriv_term

        return dz_dy

    def sag(self, x: be.ndarray, y: be.ndarray) -> be.ndarray:
        """Calculate the sag z(x, y) of the toroidal surface."""
        x2 = x**2
        z_y = self._calculate_zy(y)
        R = self.R_rot

        # Calculate base toroidal sag z = R - sqrt((R - z_y)^2 - x^2)
        if be.isinf(R):
            z = z_y
        else:
            term_inside_sqrt = (R - z_y) ** 2 - x2

            z = be.where(term_inside_sqrt < 0, be.nan, R - be.sqrt(term_inside_sqrt))

        return z

    def _surface_normal(
        self, x: be.ndarray, y: be.ndarray
    ) -> tuple[be.ndarray, be.ndarray, be.ndarray]:
        """Calculate the surface normal vector (nx, ny, nz)
        using Optiland convention."""
        z_y = self._calculate_zy(y)
        dz_dy = self._calculate_zy_derivative(y)
        R = self.R_rot

        # Partial derivatives of the toroidal part: dz/dx, dz/dy
        if be.isinf(R):
            # Cylinder extruded along X: z = z_y(y)
            fx = be.zeros_like(x)
            fy = dz_dy
            term_inside_sqrt = be.inf
        else:
            term_inside_sqrt = (R - z_y) ** 2 - x**2
            #
            valid_mask = term_inside_sqrt >= 0
            safe_term_inside_sqrt = be.where(valid_mask, term_inside_sqrt, self.eps)
            sqrt_term = be.sqrt(safe_term_inside_sqrt)
            safe_sqrt_term = be.where(be.abs(sqrt_term) < self.eps, self.eps, sqrt_term)

            fx = be.where(valid_mask, x / safe_sqrt_term, 0.0)
            fy = be.where(valid_mask, (R - z_y) * dz_dy / safe_sqrt_term, 0.0)

        # No Zernike derivatives added in this simplified version

        # Normalize according to Optiland convention: (fx, fy, -1) / mag
        mag_sq = fx**2 + fy**2 + 1.0
        mag = be.sqrt(mag_sq)
        safe_mag = be.where(mag < self.eps, 1.0, mag)

        nx = fx / safe_mag
        ny = fy / safe_mag
        nz = -1.0 / safe_mag

        # Return (0,0,-1) for invalid domain points from toroidal part.
        nx = be.where(term_inside_sqrt >= 0, nx, 0.0)
        ny = be.where(term_inside_sqrt >= 0, ny, 0.0)
        nz = be.where(term_inside_sqrt >= 0, nz, -1.0)

        return nx, ny, nz

    def __str__(self) -> str:
        return "Toroidal"

    def to_dict(self) -> dict:
        """Converts the geometry to a dictionary."""
        geometry_dict = super().to_dict()
        # Add toroidal specific parameters, remove conflicting base keys
        geometry_dict.update(
            {
                "geometry_type": self.__str__(),
                "radius_rotation": self.R_rot,
                "radius_yz": self.R_yz,
                "conic_yz": self.k_yz,
                "coeffs_poly_y": self.coeffs_poly_y.tolist()
                if hasattr(self.coeffs_poly_y, "tolist")
                else self.coeffs_poly_y,
            }
        )
        # Remove base class keys not relevant or potentially confusing here
        if "radius" in geometry_dict:
            del geometry_dict["radius"]
        if "conic" in geometry_dict:
            del geometry_dict["conic"]
        if "coefficients" in geometry_dict:
            del geometry_dict["coefficients"]  # Use specific coeffs_poly_y
        if "norm_radius" in geometry_dict:
            del geometry_dict["norm_radius"]  # No Zernike

        return geometry_dict

    @classmethod
    def from_dict(cls, data: dict) -> "ToroidalGeometry":
        """Creates a ToroidalGeometry from a dictionary representation."""
        required_keys = {"cs", "radius_rotation", "radius_yz"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required ToroidalGeometry keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(
            coordinate_system=cs,
            radius_rotation=data["radius_rotation"],
            radius_yz=data["radius_yz"],
            conic=data.get("conic_yz", 0.0),  # Match key used in to_dict
            coeffs_poly_y=data.get("coeffs_poly_y", []),
            tol=data.get("tol", 1e-10),
            max_iter=data.get("max_iter", 100),
        )
