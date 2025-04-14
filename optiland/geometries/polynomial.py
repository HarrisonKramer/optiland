"""Polynomial XY Geometry

The Polynomial XY geometry represents a surface defined by a polynomial in two
dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Cij * x^i * y^j)

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Cij are the polynomial coefficients

The coefficients are defined in a 2D array where coefficients[i][j] is the
coefficient for x^i * y^j.

Historically, XY-polynomials were the first type of polynomials used
for low-order freeform surfaces (see https://doi.org/10.1364/AO.38.003572).
These polynomials remain common surface descriptors of freeform surfaces;
however their lack of orthogonality renders them less desirable than
their orthogonal counterparts for highly corrected imaging systems.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class PolynomialGeometry(NewtonRaphsonGeometry):
    """Represents a polynomial geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Cij * x^i * y^j)

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Cij are the polynomial coefficients

    The coefficients are defined in a 2D array where coefficients[i][j] is the
    coefficient for x^i * y^j.

    Historically, XY-polynomials were the first type of polynomials used
    for low-order freeform surfaces (see https://doi.org/10.1364/AO.38.003572).
    These polynomials remain common surface descriptors of freeform surfaces;
    however their lack of orthogonality renders them less desirable than
    their orthogonal counterparts for highly corrected imaging systems.

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
            polynomial surface. Defaults to an empty list, indicating no
            polynomial coefficients are used.

    """

    def __init__(
        self,
        coordinate_system,
        radius,
        conic=0.0,
        tol=1e-10,
        max_iter=100,
        coefficients=None,
    ):
        if coefficients is None:
            coefficients = []
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.c = be.atleast_2d(coefficients)
        self.is_symmetric = False

        if len(self.c) == 0:
            self.c = be.zeros((1, 1))

    def __str__(self):
        return "Polynomial XY"

    def sag(self, x=0, y=0):
        """Calculates the sag of the polynomial surface at the given coordinates.

        Args:
            x (float, be.ndarray, optional): The x-coordinate(s).
                Defaults to 0.
            y (float, be.ndarray, optional): The y-coordinate(s).
                Defaults to 0.

        Returns:
            float: The sag value at the given coordinates.

        """
        r2 = x**2 + y**2
        z = r2 / (self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))
        for i in range(len(self.c)):
            for j in range(len(self.c[i])):
                z += self.c[i][j] * (x**i) * (y**j)
        return z

    def _surface_normal(self, x, y):
        """Calculates the surface normal of the polynomial surface at the given x
        and y position.

        Args:
            x (be.ndarray): The x values to use for calculation.
            y (be.ndarray): The y values to use for calculation.

        Returns:
            tuple: The surface normal components (nx, ny, nz).

        """
        r2 = x**2 + y**2
        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dzdx = x / denom
        dzdy = y / denom

        for i in range(1, len(self.c)):
            for j in range(len(self.c[i])):
                dzdx += i * self.c[i][j] * (x ** (i - 1)) * (y**j)

        for i in range(len(self.c)):
            for j in range(1, len(self.c[i])):
                dzdy += j * self.c[i][j] * (x**i) * (y ** (j - 1))

        norm = be.sqrt(dzdx**2 + dzdy**2 + 1)
        nx = dzdx / norm
        ny = dzdy / norm
        nz = -1 / norm

        return nx, ny, nz

    def to_dict(self):
        """Converts the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict["coefficients"] = self.c.tolist()
        return geometry_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a PolynomialGeometry from a dictionary.

        Args:
            data (dict): The dictionary containing the geometry data.

        Returns:
            PolynomialGeometry: The geometry created from the dictionary.

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
        )
