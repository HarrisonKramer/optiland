"""Chebyshev Geometry

The Chebyshev polynomial geometry represents a surface defined by a Chebyshev
polynomial in two dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
    sum(Cij * T_i(x / norm_x) * T_j(y / norm_y))

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Cij are the Chebyshev polynomial coefficients
- T_i(x) is the Chebyshev polynomial of the first kind of degree i
- norm_x and norm_y are normalization factors for the x and y coordinates

Chebyshev polynomials are derived in Cartesian coordinates,
which - unlike many other polynomial freeform surfaces used
to describe rotationally-symmetric systems - allows for
straightforward definition of anamorphic or non-rotationally
symmetric systems and non-elliptical apertures.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class ChebyshevPolynomialGeometry(NewtonRaphsonGeometry):
    """Represents a Chebyshev polynomial geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) +
        sum(Cij * T_i(x / norm_x) * T_j(y / norm_y))

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Cij are the Chebyshev polynomial coefficients
    - T_i(x) is the Chebyshev polynomial of the first kind of degree i
    - norm_x and norm_y are normalization factors for the x and y coordinates

    The coefficients are defined in a 2D array where coefficients[i][j] is the
    coefficient for T_i(x) * T_j(y).

    Chebyshev polynomials are derived in Cartesian coordinates,
    which - unlike many other polynomial freeform surfaces used
    to describe rotationally-symmetric systems - allows for
    straightforward definition of anamorphic or non-rotationally
    symmetric systems and non-elliptical apertures.

    Args:
        coordinate_system (str): The coordinate system used for the geometry.
        radius (float): The radius of curvature of the geometry.
        conic (float, optional): The conic constant of the geometry.
            Defaults to 0.0.
        tol (float, optional): The tolerance value used in calculations.
            Defaults to 1e-10.
        max_iter (int, optional): The maximum number of iterations used in
            calculations. Defaults to 100.
        coefficients (list or be.ndarray, optional): The coefficients of the
            Chebyshev polynomial surface. Defaults to an empty list, indicating
            no Chebyshev polynomial coefficients are used.
        norm_x (int, optional): The normalization factor for the x-coordinate.
            Defaults to 1.
        norm_y (int, optional): The normalization factor for the y-coordinate.
            Defaults to 1.

    """

    def __init__(
        self,
        coordinate_system,
        radius,
        conic=0.0,
        tol=1e-10,
        max_iter=100,
        coefficients=None,
        norm_x=1,
        norm_y=1,
    ):
        if coefficients is None:
            coefficients = []
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.c = be.atleast_2d(coefficients)
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.is_symmetric = False

    def __str__(self):
        return "Chebyshev Polynomial"

    def sag(self, x=0, y=0):
        """Calculates the sag of the Chebyshev polynomial surface at the given
        coordinates.

        Args:
            x (float, be.ndarray, optional): The x-coordinate(s).
                Defaults to 0.
            y (float, be.ndarray, optional): The y-coordinate(s).
                Defaults to 0.

        Returns:
            float: The sag value at the given coordinates.

        """
        x_norm = x / self.norm_x
        y_norm = y / self.norm_y

        self._validate_inputs(x_norm, y_norm)

        r2 = x**2 + y**2
        z = r2 / (self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))

        non_zero_indices = be.argwhere(self.c != 0)
        for i, j in non_zero_indices:
            z += self.c[i, j] * self._chebyshev(i, x_norm) * self._chebyshev(j, y_norm)

        return z

    def _surface_normal(self, x, y):
        """Calculates the surface normal of the Chebyshev polynomial surface at
        the given x and y position.

        Args:
            x (be.ndarray): The x values to use for calculation.
            y (be.ndarray): The y values to use for calculation.

        Returns:
            tuple: The surface normal components (nx, ny, nz).

        """
        x_norm = x / self.norm_x
        y_norm = y / self.norm_y

        self._validate_inputs(x_norm, y_norm)

        r2 = x**2 + y**2
        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dzdx = x / denom
        dzdy = y / denom

        non_zero_indices = be.argwhere(self.c != 0)
        for i, j in non_zero_indices:
            dzdx += (
                self._chebyshev_derivative(i, x_norm)
                * self.c[i, j]
                * self._chebyshev(j, y_norm)
            )
            dzdy += (
                self._chebyshev_derivative(j, y_norm)
                * self.c[i, j]
                * self._chebyshev(i, x_norm)
            )

        norm = be.sqrt(dzdx**2 + dzdy**2 + 1)
        nx = dzdx / norm
        ny = dzdy / norm
        nz = -1 / norm

        return nx, ny, nz

    def _chebyshev(self, n, x):
        """Calculates the Chebyshev polynomial of the first kind of degree n at
        the given x value.

        Args:
            n (int): The degree of the Chebyshev polynomial.
            x (be.ndarray): The x value to use for calculation.

        Returns:
            be.ndarray: The Chebyshev polynomial of the first kind of degree n
                at the given x value.

        """
        return be.cos(n * be.arccos(x))

    def _chebyshev_derivative(self, n, x):
        """Calculates the derivative of the Chebyshev polynomial of the first kind
        of degree n at the given x value.

        Args:
            n (int): The degree of the Chebyshev polynomial.
            x (be.ndarray): The x value to use for calculation.

        Returns:
            be.ndarray: The derivative of the Chebyshev polynomial of the first
                kind of degree n at the given x value.

        """
        return n * be.sin(n * be.arccos(x)) / be.sqrt(1 - x**2)

    def _validate_inputs(self, x_norm, y_norm):
        """Validates the input coordinates for the Chebyshev polynomial surface.

        Args:
            x_norm (be.ndarray): The normalized x values.
            y_norm (be.ndarray): The normalized y values.

        """
        if be.any(be.abs(x_norm) > 1) or be.any(be.abs(y_norm) > 1):
            raise ValueError(
                "Chebyshev input coordinates must be normalized "
                "to [-1, 1]. Consider updating the normalization "
                "factors.",
            )

    def to_dict(self):
        """Converts the Chebyshev polynomial geometry to a dictionary.

        Returns:
            dict: The Chebyshev polynomial geometry as a dictionary.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update(
            {
                "coefficients": self.c.tolist(),
                "norm_x": self.norm_x,
                "norm_y": self.norm_y,
            },
        )

        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a Chebyshev polynomial geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the Chebyshev
                polynomial geometry.

        Returns:
            ChebyshevPolynomialGeometry: The Chebyshev polynomial geometry.

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
            data.get("norm_x", 1),
            data.get("norm_y", 1),
        )
