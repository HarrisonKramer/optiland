"""Odd Asphere Geometry

The Even Asphere geometry represents a surface defined by an even asphere in
two dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Ci * r^i)

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Ci are the aspheric coefficients

Kramer Harrison, 2025
"""

import warnings

import optiland.backend as be
from optiland.geometries.even_asphere import EvenAsphere


class OddAsphere(EvenAsphere):
    """Represents an odd asphere geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Ci * r^i)

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Ci are the aspheric coefficients
    - i is an integer from 1 to n

    Args:
        coordinate_system (str): The coordinate system used for the geometry.
        radius (float): The radius of curvature of the geometry.
        conic (float, optional): The conic constant of the geometry.
            Defaults to 0.0.
        tol (float, optional): The tolerance value used in calculations.
            Defaults to 1e-10.
        max_iter (int, optional): The maximum number of iterations used in
            calculations. Defaults to 100.
        coefficients (list, optional): The coefficients of the asphere.
            Defaults to an empty list, indicating no aspheric coefficients are
            used.

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
        super().__init__(coordinate_system, radius, conic, tol, max_iter, coefficients)
        self.order = 1  # used for optimization scaling

    def __str__(self):
        return "Odd Asphere"

    def sag(self, x=0, y=0):
        """Calculates the sag of the asphere at the given coordinates.

        Args:
            x (float, be.ndarray, optional): The x-coordinate(s).
                Defaults to 0.
            y (float, be.ndarray, optional): The y-coordinate(s).
                Defaults to 0.

        Returns:
            float: The sag value at the given coordinates.

        """
        r2 = x**2 + y**2
        r = be.sqrt(r2)
        z = r2 / (self.radius * (1 + be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))
        for i, Ci in enumerate(self.c):
            z += Ci * r ** (i + 1)

        return z

    def _surface_normal(self, x, y):
        """Calculates the surface normal of the asphere at the given x and y
        position.

        Args:
            x (be.ndarray): The x values to use for calculation.
            y (be.ndarray): The y values to use for calculation.

        Returns:
            tuple: The surface normal components (nx, ny, nz).

        """
        r2 = x**2 + y**2
        r = be.sqrt(r2)

        denom = self.radius * be.sqrt(1 - (1 + self.k) * r2 / self.radius**2)
        dfdx = x / denom
        dfdy = y / denom

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, Ci in enumerate(self.c):
                x_term = (i + 1) * x * Ci * r ** (i - 1)
                y_term = (i + 1) * y * Ci * r ** (i - 1)

                x_term[~be.isfinite(x_term)] = 0
                y_term[~be.isfinite(y_term)] = 0

                dfdx += x_term
                dfdy += y_term

        mag = be.sqrt(dfdx**2 + dfdy**2 + 1)

        nx = dfdx / mag
        ny = dfdy / mag
        nz = -1 / mag

        return nx, ny, nz
