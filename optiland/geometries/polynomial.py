import numpy as np
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class PolynomialGeometry(NewtonRaphsonGeometry):
    """
    Represents a polynomial geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Cij * x^i * y^j)

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Cij are the polynomial coefficients

    The coefficients are defined in a 2D array where coefficients[i][j] is the
    coefficient for x^i * y^j.

    Args:
        coordinate_system (str): The coordinate system used for the geometry.
        radius (float): The radius of curvature of the geometry.
        conic (float, optional): The conic constant of the geometry.
            Defaults to 0.0.
        tol (float, optional): The tolerance value used in calculations.
            Defaults to 1e-10.
        max_iter (int, optional): The maximum number of iterations used in
            calculations. Defaults to 100.
        coefficients (list or np.ndarray, optional): The coefficients of the
            polynomial surface. Defaults to an empty list, indicating no
            polynomial coefficients are used.
    """

    def __init__(self, coordinate_system, radius, conic=0.0,
                 tol=1e-10, max_iter=100, coefficients=[]):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.c = coefficients
        self.is_symmetric = False

        if len(self.c) == 0:
            self.c = np.zeros((1, 1))

    def sag(self, x=0, y=0):
        """
        Calculates the sag of the polynomial surface at the given coordinates.

        Args:
            x (float, np.ndarray, optional): The x-coordinate(s).
                Defaults to 0.
            y (float, np.ndarray, optional): The y-coordinate(s).
                Defaults to 0.

        Returns:
            float: The sag value at the given coordinates.
        """
        r2 = x**2 + y**2
        z = r2 / (self.radius *
                  (1 + np.sqrt(1 - (1 + self.k) * r2 / self.radius**2)))
        for i in range(len(self.c)):
            for j in range(len(self.c[i])):
                z += self.c[i][j] * (x ** i) * (y ** j)
        return z

    def _surface_normal(self, x, y):
        """
        Calculates the surface normal of the polynomial surface at the given x
        and y position.

        Args:
            x (np.ndarray): The x values to use for calculation.
            y (np.ndarray): The y values to use for calculation.

        Returns:
            tuple: The surface normal components (nx, ny, nz).
        """
        r2 = x**2 + y**2
        denom = self.radius * np.sqrt(1 - (1 + self.k)*r2 / self.radius**2)
        dzdx = x / denom
        dzdy = y / denom

        for i in range(1, len(self.c)):
            for j in range(len(self.c[i])):
                dzdx += i * self.c[i][j] * (x ** (i - 1)) * (y ** j)

        for i in range(len(self.c)):
            for j in range(1, len(self.c[i])):
                dzdy += j * self.c[i][j] * (x ** i) * (y ** (j - 1))

        norm = np.sqrt(dzdx**2 + dzdy**2 + 1)
        nx = dzdx / norm
        ny = dzdy / norm
        nz = -1 / norm

        return nx, ny, nz
