"""Chebyshev Coefficients Variable Module

This module contains the class for a Chebyshev coefficient variable in an optic
system. The ChebyshevCoeffVariable class is a subclass of the
PolynomialCoeffVariable class that represents a variable for a Chebyshev
coefficient of a ChebyshevGeometry. It is used in the optimization process for
Chebyshev geometries.

Kramer Harrison, 2024
"""

from optiland.optimization.variable.polynomial_coeff import PolynomialCoeffVariable


class ChebyshevCoeffVariable(PolynomialCoeffVariable):
    """Represents a variable for a Chebyshev coefficient of a ChebyshevGeometry.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The index of the surface in the optical system.
        coeff_index (tuple(int, int)): The (x, y) indices of the Chebyshev
            coefficient.
        apply_scaling (bool): Whether to apply scaling to the variable.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Attributes:
        coeff_number (int): The index of the Chebyshev coefficient.

    """

    def __init__(
        self,
        optic,
        surface_number,
        coeff_index,
        apply_scaling=True,
        **kwargs,
    ):
        super().__init__(optic, surface_number, coeff_index, apply_scaling, **kwargs)

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Chebyshev Coeff. {self.coeff_index}, Surface {self.surface_number}"
