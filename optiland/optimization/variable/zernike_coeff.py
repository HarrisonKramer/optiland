"""Zernike Coefficients Variable Module

This module contains the class for a Zernike coefficient variable in an optic
system. The ZernikeCoeffVariable class is a subclass of the
PolynomialCoeffVariable class that represents a variable for a Zernike
coefficient of a ZernikeGeometry. It is used in the optimization process for
Zernike geometries.

drpaprika, 2025
"""

from __future__ import annotations

import numpy as np

from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable.polynomial_coeff import PolynomialCoeffVariable


class ZernikeCoeffVariable(PolynomialCoeffVariable):
    """Represents a variable for a Zernike coefficient of a ZernikeGeometry.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The index of the surface in the optical system.
        coeff_index (int): The i index of the Zernike coefficient.
        scaler (Scaler): The scaler to use for the variable. Defaults to
            IdentityScaler().
        **kwargs: Additional keyword arguments.

    Attributes:
        coeff_number (int): The index of the Zernike coefficient.

    """

    def __init__(
        self,
        optic,
        surface_number,
        coeff_index,
        scaler=None,
        **kwargs,
    ):
        if scaler is None:
            scaler = IdentityScaler()
        super().__init__(optic, surface_number, coeff_index, scaler=scaler, **kwargs)

    def get_value(self):
        """Get the current value of the Zernike coefficient.

        Returns:
            float: The current value of the Zernike coefficient.

        """
        surf = self._surfaces.surfaces[self.surface_number]
        i = self.coeff_index
        try:
            value = surf.geometry.coefficients[i]
        except IndexError:
            pad_width_i = max(0, i + 1)
            c_new = np.pad(
                surf.geometry.coefficients,
                pad_width=(0, pad_width_i),
                mode="constant",
                constant_values=0,
            )
            surf.geometry.coefficients = c_new
            value = 0
        return value

    def update_value(self, new_value):
        """Update the value of the Zernike coefficient.

        Args:
            new_value (float): The new value of the Zernike coefficient.

        """
        surf = self.optic.surface_group.surfaces[self.surface_number]
        i = self.coeff_index

        if i < len(surf.geometry.coefficients):
            # If the coefficient already exists, update it
            surf.geometry.coefficients[i] = new_value
        else:
            # If the coefficient does not exist, pad the array and set the value
            pad_width_i = max(0, i + 1)
            new_coefficients = np.pad(
                surf.geometry.coefficients,
                pad_width=(0, pad_width_i),
                mode="constant",
                constant_values=0,
            )
            new_coefficients[i] = new_value
            surf.geometry.coefficients = new_coefficients

    def __str__(self):
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Zernike Coeff. {self.coeff_index}, Surface {self.surface_number}"
