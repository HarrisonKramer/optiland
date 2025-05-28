"""Zernike Geometry

The Zernike polynomial geometry represents a surface defined by a Zernike
polynomial in two dimensions. The surface is defined as:

z(x,y) = z_{base}(x,y) + r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
    sum_i [c[i] * Z_i(rho, theta)]

where:
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- c[i] is the coefficient for the i-th Fringe Zernike polynomial
- Z_i(...) is the i-th Fringe Zernike polynomial in polar coordinates
- rho = sqrt(x^2 + y^2) / normalization, theta = atan2(y, x)

Zernike polynomials are a set of orthogonal functions defined over the unit
disk, widely used in freeform optical surface design. They efficiently
describe wavefront aberrations and complex surface deformations by decomposing
them into radial and azimuthal components. Their orthogonality ensures minimal
cross-coupling between terms, making them ideal for optimizing optical systems.
In freeform optics, they enable precise control of surface shape,
improving performance beyond traditional spherical and aspheric designs.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class ZernikePolynomialGeometry(NewtonRaphsonGeometry):
    """Represents a Fringe Zernike polynomial geometry defined as:

    z(x,y) = z_{base}(x,y) + r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
        sum_i [c[i] * Z_i(rho, theta)]

    where:
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - c[i] is the coefficient for the i-th Fringe Zernike polynomial
    - Z_i(...) is the i-th Fringe Zernike polynomial in polar coordinates
    - rho = sqrt(x^2 + y^2) / normalization, theta = atan2(y, x)

    The coefficients are defined in a 1D array where coefficients[i] is the
    coefficient for Z_i.

    Args:
        coordinate_system (CoordinateSystem): The coordinate system of the geometry.
        radius (float): The radius of curvature of the base sphere.
        conic (float, optional): The conic constant of the base sphere.
            Defaults to 0.0.
        tol (float, optional): Tolerance for Newton-Raphson iteration.
            Defaults to 1e-10.
        max_iter (int, optional): Maximum iterations for Newton-Raphson.
            Defaults to 100.
        coefficients (list[float] or be.ndarray, optional): A 1D array or list
            of Fringe Zernike coefficients c_i. `coefficients[i]` is the
            coefficient for Z_(i+1). Defaults to an empty list.
        norm_radius (float, optional): Normalization radius for rho.
            Defaults to 1.0.

    Attributes:
        c (be.ndarray): 1D array of Fringe Zernike coefficients.
        norm_radius (float): Normalization radius.

    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,  # Corrected type
        radius: float,
        conic: float = 0.0,
        tol: float = 1e-10,
        max_iter: int = 100,
        coefficients: list[float] or be.ndarray = None,  # Allow list or be.ndarray
        norm_radius: float = 1.0,  # Default to float
    ):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        if coefficients is None:
            coefficients = []
        self.c = be.atleast_1d(be.asarray(coefficients))  # Ensure it's a backend array
        self.norm_radius = float(norm_radius)  # Ensure norm_radius is float
        self.is_symmetric = False  # Zernikes can be non-symmetric

    def __str__(self) -> str:
        return "Zernike Polynomial"

    def sag(
        self, x: float or be.ndarray, y: float or be.ndarray
    ) -> float or be.ndarray:
        """Calculate the sag of the Zernike polynomial surface at the given
        coordinates.

        Args:
            x (float or be.ndarray): The Cartesian x-coordinate(s).
            y (float or be.ndarray): The Cartesian y-coordinate(s).

        Returns:
            be.ndarray or float: The sag value at the given Cartesian coordinates.

        """
        x_norm = x / self.norm_radius
        y_norm = y / self.norm_radius

        self._validate_inputs(x_norm, y_norm)

        # Convert to local polar
        rho = be.sqrt(x_norm**2 + y_norm**2)
        theta = be.arctan2(y_norm, x_norm)

        # Base conic
        r2 = x**2 + y**2
        z = r2 / (self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))

        # Add normalized Fringe Zernike contributions
        # Sum over all nonzero coefficients
        non_zero_indices = be.nonzero(self.c)[0]
        for i in non_zero_indices:
            normalization_factor = be.sqrt(2 * (i + 1) / be.pi)
            z_i = self._zernike(i + 1, rho, theta)
            z = z + normalization_factor * self.c[i] * z_i

        return z

    def _zernike(self, i: int, rho: be.ndarray, theta: be.ndarray) -> be.ndarray:
        """Calculate the i-th Fringe Zernike polynomial at the given rho, theta.

        Args:
            i (int): Fringe Zernike index (1-based).
            rho (be.ndarray or float): Radial coordinate (normalized).
            theta (be.ndarray or float): Azimuthal coordinate (radians).

        Returns:
            be.ndarray or float: Z_i(rho, theta) value(s).

        """
        n, m = self._fringezernike_order_to_zernike_order(i)
        Rnm = self._radial_poly(n, abs(m), rho)

        if m == 0:
            return Rnm
        if m > 0:
            return Rnm * be.cos(m * theta)
        return Rnm * be.sin(abs(m) * theta)

    def _zernike_derivative(
        self,
        i: int,
        rho: be.ndarray,
        theta: be.ndarray,
    ) -> tuple[be.ndarray, be.ndarray]:
        """Return partial derivatives of Z_i w.r.t. rho and theta:
        (dZ/drho, dZ/dtheta).
        We'll use them in chain rule for partial derivatives w.r.t x,y.

        Args:
            i (int): Fringe Zernike index (1-based).
            rho (be.ndarray or float): Radial coordinate (normalized).
            theta (be.ndarray or float): Azimuthal coordinate (radians).

        Returns:
            tuple[be.ndarray or float, be.ndarray or float]: Partial derivatives
            (dZ/drho, dZ/dtheta).

        """
        n, m = self._fringezernike_order_to_zernike_order(i)
        Rnm = self._radial_poly(n, abs(m), rho)
        dRnm = self._radial_poly_derivative(n, abs(m), rho)

        if m == 0:
            # Z_n^0(rho,theta) = R_n^0(rho), no theta dependence
            dZdrho = dRnm
            dZdtheta = 0.0
        elif m > 0:
            # Z_n^m = R_n^m(rho)*cos(m*theta)
            # d/d rho -> dR_n^m(rho)*cos(m*theta)
            dZdrho = dRnm * be.cos(m * theta)
            # d/d theta -> R_n^m(rho)*(-m sin(m theta))
            dZdtheta = -m * Rnm * be.sin(m * theta)
        else:
            # m < 0 => Z_n^m = R_n^|m|(rho)*sin(|m|*theta)
            dZdrho = dRnm * be.sin(abs(m) * theta)
            dZdtheta = abs(m) * Rnm * be.cos(abs(m) * theta)

        return dZdrho, dZdtheta

    def _radial_poly(self, n: int, m: int, rho: be.ndarray) -> be.ndarray:
        """Compute the radial polynomial R_n^m(rho).

        R_n^m(rho) = sum_{k=0}^{(n - m)/2} (-1)^k * (n-k)! /
                     [k! ((n+m)/2 - k)! ((n-m)/2 - k)!] * rho^(n - 2k)

        Args:
            n (int): Radial Zernike order (n >= 0).
            m (int): Azimuthal Zernike order (m >= 0, |m| <= n, n-m is even).
            rho (be.ndarray or float): Radial coordinate (normalized).

        Returns:
            be.ndarray or float: The radial polynomial R_n^m(rho) value(s).

        """
        val = 0.0
        upper_k = (n - m) // 2
        for k in range(upper_k + 1):
            sign = (-1) ** k
            numerator = factorial(n - k)
            denominator = (
                factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
            )
            val = val + sign * (numerator / denominator) * (rho ** (n - 2 * k))
        return val

    def _radial_poly_derivative(self, n: int, m: int, rho: be.ndarray):
        """Derivative of the radial polynomial R_n^m(rho) with respect to rho.

        d/d(rho) R_n^m(rho) = sum_{k=0} (...) (n-2k) * rho^(n-2k-1)

        Args:
            n (int): Radial Zernike order.
            m (int): Azimuthal Zernike order (m >= 0).
            rho (be.ndarray or float): Radial coordinate (normalized).

        Returns:
            be.ndarray or float: Derivative d(R_n^m)/d(rho) value(s).

        """
        val = 0.0
        upper_k = (n - m) // 2
        for k in range(upper_k + 1):
            sign = (-1) ** k
            numerator = factorial(n - k)
            denominator = (
                factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
            )
            factor = n - 2 * k
            if factor < 0:
                continue
            power_term = rho ** (n - 2 * k - 1) if (n - 2 * k - 1) >= 0 else 0
            val = val + sign * (numerator / denominator) * factor * power_term
        return val

    def _surface_normal(
        self,
        x: be.ndarray,
        y: be.ndarray,
    ) -> tuple[float, float, float]:
        """Calculate the surface normal of the full surface (conic + Zernike)
        in Cartesian coordinates at (x, y).

        Args:
            x (be.ndarray): x-coordinate(s) on the surface.
            y (be.ndarray): y-coordinate(s) on the surface.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: Normal vector components
            (nx, ny, nz) in Cartesian coordinates.

        """
        # Conic partial derivatives:
        r2 = x**2 + y**2
        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dzdx = x / denom
        dzdy = y / denom

        # Protect against divide-by-zero for r=0
        # or handle small r if needed
        eps = 1e-14
        denom = be.where(be.abs(denom) < eps, eps, denom)

        # Now add partial derivatives from the Zernike expansions
        x_norm = x / self.norm_radius
        y_norm = y / self.norm_radius
        rho = be.sqrt(x_norm**2 + y_norm**2)
        theta = be.arctan2(y_norm, x_norm)

        # Chain rule:
        # dZ/dx = dZ/drho * d(rho)/dx + dZ/dtheta * d(theta)/dx
        # We'll define the partials of (rho,theta) wrt x:
        #   drho/dx    = x / (norm_x^2 * rho)
        #   dtheta/dx  = - y / (rho^2 * norm_y * norm_x)
        drho_dx = (
            be.zeros_like(x)
            if be.all(rho == 0)
            else ((x / (self.norm_radius**2)) / (rho + eps))
        )
        drho_dy = (
            be.zeros_like(y)
            if be.all(rho == 0)
            else ((y / (self.norm_radius**2)) / (rho + eps))
        )
        dtheta_dx = -(y_norm) / (rho**2 + eps) * (1.0 / self.norm_radius)
        dtheta_dy = +(x_norm) / (rho**2 + eps) * (1.0 / self.norm_radius)

        non_zero_indices = be.nonzero(self.c)[0]
        for i in non_zero_indices:
            dZdrho, dZdtheta = self._zernike_derivative(i, rho, theta)
            # partial wrt x
            dzdx = dzdx + self.c[i] * (dZdrho * drho_dx + dZdtheta * dtheta_dx)
            # partial wrt y
            dzdy = dzdy + self.c[i] * (dZdrho * drho_dy + dZdtheta * dtheta_dy)

        # Surface normal vector in cartesian coords: (-dzdx, -dzdy, 1)
        # normalized. Check sign conventions!
        nx = +dzdx
        ny = +dzdy
        norm = be.sqrt(nx**2 + ny**2 + 1)
        norm = be.where(norm < eps, 1.0, norm)  # Avoid division by zero
        nx = nx / norm
        ny = ny / norm
        nz = -be.ones_like(x) / norm

        return (nx, ny, nz)

    @staticmethod
    def _fringezernike_order_to_zernike_order(k: int) -> tuple[float, float]:
        """Convert Fringe Zernike index k to classical Zernike (n, m).

        https://wp.optics.arizona.edu/visualopticslab/wp-content/
        uploads/sites/52/2021/10/Zernike-Fit.pdf
        Note: k is 0-indexed in the paper, but 1-indexed for Fringe Zernikes.
        The formula here assumes k is the 1-based Fringe index.
        """
        # Adjust k to be 0-indexed for the formula if it's passed as 1-indexed Fringe
        k_zero_indexed = k - 1
        n = be.ceil((-3 + be.sqrt(9 + 8 * k_zero_indexed)) / 2)
        m = 2 * k_zero_indexed - n * (n + 2)
        return (n.astype(int), m.astype(int))

    def _validate_inputs(
        self, x_norm: be.ndarray or float, y_norm: be.ndarray or float
    ) -> None:
        """Validate the input coordinates for the Zernike polynomial surface.

        Args:
            x_norm (be.ndarray or float): The normalized x-coordinate(s).
            y_norm (be.ndarray or float): The normalized y-coordinate(s).

        """
        if be.any(be.abs(x_norm) > 1) or be.any(be.abs(y_norm) > 1):
            raise ValueError(
                "Zernike coordinates must be normalized "
                "to [-1, 1]. Consider updating the normalization "
                "radius to 1.1x the surface aperture.",
            )

    def to_dict(self) -> dict:
        """Convert the Zernike polynomial geometry to a dictionary.

        Returns:
            dict: A dictionary representation of the Zernike polynomial geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "coefficients": self.c.tolist(),
                "norm_radius": self.norm_radius,
            },
        )

        return geometry_dict

    @classmethod
    def from_dict(cls, data: dict) -> "ZernikePolynomialGeometry":
        """Create a Zernike polynomial geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the Zernike
                polynomial geometry.

        Returns:
            ZernikePolynomialGeometry: The Zernike polynomial geometry.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(
            cs,
            data["radius"],
            data.get("conic", 0.0),
            data.get("tol", 1e-10),
            data.get("max_iter", 100),
            data.get("coefficients", []),
            data.get("coefficients", []),
            data.get("norm_radius", 1.0),  # Corrected from norm_x, norm_y
        )


def factorial(n: int) -> int or float:  # Type hint for n
    """Computes factorial of n.

    Args:
        n (int): Non-negative integer.

    Returns:
        int or float: n!
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0:
        return 1
    # Use be.prod for backend compatibility, be.arange needs to be cast if not float
    return be.prod(be.arange(1, n + 1, dtype=be.int32))
