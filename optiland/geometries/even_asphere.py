"""Even Asphere Geometry

The Even Asphere geometry represents a surface defined by an even asphere in
two dimensions. The surface is defined as:

z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Ci * r^(2i))

where
- r^2 = x^2 + y^2
- R is the radius of curvature
- k is the conic constant
- Ci are the aspheric coefficients

Even-order aspheric surfaces are commonly used to correct specific aberrations
while maintaining rotational symmetry. These surfaces are defined by polynomial
terms with even exponents, ensuring symmetry about the optical axis.

Kramer Harrison, 2024
"""
import numpy as np
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry
from optiland.coordinate_system import CoordinateSystem


class EvenAsphere(NewtonRaphsonGeometry):
    """
    Represents an even asphere geometry defined as:

    z = r^2 / (R * (1 + sqrt(1 - (1 + k) * r^2 / R^2))) + sum(Ci * r^(2i))

    where
    - r^2 = x^2 + y^2
    - R is the radius of curvature
    - k is the conic constant
    - Ci are the aspheric coefficients

    Even-order aspheric surfaces are commonly used to correct specific
    aberrations while maintaining rotational symmetry. These surfaces are
    defined by polynomial terms with even exponents, ensuring symmetry about
    the optical axis.

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

    def __init__(self, coordinate_system, radius, conic=0.0,
                 tol=1e-10, max_iter=100, coefficients=[]):
        super().__init__(coordinate_system, radius, conic, tol, max_iter)
        self.c = coefficients
        self.is_symmetric = True
        self.order = 2  # used for optimization scaling

    def __str__(self):
        return 'Even Asphere'

    def sag(self, x=0, y=0):
        """
        Calculates the sag of the asphere at the given coordinates.

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
        for i, Ci in enumerate(self.c):
            z += Ci * r2 ** (i + 1)

        return z

    def _surface_normal(self, x, y):
        """
        Calculates the surface normal of the asphere at the given x and y
        position.

        Args:
            x (np.ndarray): The x values to use for calculation.
            y (np.ndarray): The y values to use for calculation.

        Returns:
            tuple: The surface normal components (nx, ny, nz).
        """
        r2 = x**2 + y**2

        denom = self.radius * np.sqrt(1 - (1 + self.k)*r2 / self.radius**2)
        dfdx = x / denom
        dfdy = y / denom

        for i, Ci in enumerate(self.c):
            dfdx += 2 * (i+1) * x * Ci * r2**i
            dfdy += 2 * (i+1) * y * Ci * r2**i

        mag = np.sqrt(dfdx**2 + dfdy**2 + 1)

        nx = dfdx / mag
        ny = dfdy / mag
        nz = -1 / mag

        return nx, ny, nz

    def to_dict(self):
        """
        Converts the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.
        """
        data = super().to_dict()
        data["coefficients"] = self.c

        return data

    @classmethod
    def from_dict(cls, data):
        """
        Creates an asphere from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the asphere.

        Returns:
            EvenAsphere: The asphere.
        """
        required_keys = {'cs', 'radius'}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data['cs'])
        conic = data.get('conic', 0.0)
        tol = data.get('tol', 1e-10)
        max_iter = data.get('max_iter', 100)
        coefficients = data.get('coefficients', [])

        return cls(cs, data['radius'],
                   conic, tol, max_iter, coefficients)
