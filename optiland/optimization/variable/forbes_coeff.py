"""This module provides specialized variable handlers for Forbes polynomial
coefficients, distinguishing between the rotationally symmetric Q-bfs
surfaces and the freeform Q-2D surfaces.

Manuel Fragata Mendes, August 2025
"""

from optiland.geometries.forbes.geometry import ForbesQ2dGeometry, ForbesQbfsGeometry
from optiland.optimization.variable.base import VariableBehavior


class ForbesQbfsCoeffVariable(VariableBehavior):
    """
    Represents a variable for a Forbes Q-bfs (rotationally symmetric) coefficient.
    This variable targets an individual coefficient `c_n` from the `coeffs_c` list.
    """

    def __init__(
        self, optic, surface_number, coeff_number, apply_scaling=False, **kwargs
    ):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)
        self.coeff_number = coeff_number

    def get_value(self):
        """Gets the value of the nth Q-bfs coefficient."""
        surf = self._surfaces.surfaces[self.surface_number]
        geom = surf.geometry
        if not isinstance(geom, ForbesQbfsGeometry):
            raise TypeError("This variable is only for ForbesQbfsGeometry.")

        coeffs = geom.coeffs_c
        value = coeffs[self.coeff_number] if self.coeff_number < len(coeffs) else 0.0

        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value):
        """Updates the value of the nth Q-bfs coefficient."""
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)

        surf = self.optic.surface_group.surfaces[self.surface_number]
        geom = surf.geometry
        if not isinstance(geom.coeffs_c, list):
            geom.coeffs_c = list(geom.coeffs_c)

        if self.coeff_number >= len(geom.coeffs_c):
            pad_width = self.coeff_number + 1 - len(geom.coeffs_c)
            geom.coeffs_c.extend([0.0] * pad_width)

        geom.coeffs_c[self.coeff_number] = new_value

    def scale(self, value):
        """
        So far, scaling is a no-op for Forbes coefficients.
        This method can be overridden if scaling is needed in the future.
        """
        scaling_factor = 1.0
        return value * scaling_factor

    def inverse_scale(self, scaled_value):
        """Inverse scales the value of the variable."""
        scaling_factor = 1.0
        return scaled_value / scaling_factor

    def __str__(self):
        return (
            f"Forbes Q-bfs Coeff n={self.coeff_number}, Surface {self.surface_number}"
        )


class ForbesQ2dCoeffVariable(VariableBehavior):
    """
    Represents a variable for a Forbes Q-2D (freeform) coefficient.
    This variable targets the coefficient `c` corresponding to a specific
    (n, m) term defined in the `coeffs_n` list.
    """

    def __init__(
        self, optic, surface_number, coeff_nm_tuple, apply_scaling=False, **kwargs
    ):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)
        if not isinstance(coeff_nm_tuple, tuple) or len(coeff_nm_tuple) != 2:
            raise ValueError("`coeff_nm_tuple` must be a tuple of (n, m).")
        self.coeff_nm_tuple = coeff_nm_tuple

    def _get_coeff_index(self, geom):
        """Finds the index of the (n, m) tuple in the geometry's coeffs_n list."""
        try:
            return geom.coeffs_n.index(self.coeff_nm_tuple)
        except (ValueError, AttributeError):
            return -1  # Not found

    def get_value(self):
        """Gets the value of the c coefficient for the specified (n, m) term."""
        surf = self._surfaces.surfaces[self.surface_number]
        geom = surf.geometry
        if not isinstance(geom, ForbesQ2dGeometry):
            raise TypeError("This variable is only for ForbesQ2dGeometry.")

        idx = self._get_coeff_index(geom)
        if idx != -1 and idx < len(geom.coeffs_c):
            return geom.coeffs_c[idx]
        return 0.0

    def update_value(self, new_value):
        """Updates the value of the c coefficient for the specified (n, m) term."""
        surf = self.optic.surface_group.surfaces[self.surface_number]
        geom = surf.geometry

        if not isinstance(geom.coeffs_n, list):
            geom.coeffs_n = list(geom.coeffs_n)
        if not isinstance(geom.coeffs_c, list):
            geom.coeffs_c = list(geom.coeffs_c)

        idx = self._get_coeff_index(geom)
        if idx != -1:
            geom.coeffs_c[idx] = new_value
        else:
            geom.coeffs_n.append(self.coeff_nm_tuple)
            geom.coeffs_c.append(new_value)

    def __str__(self):
        n, m = self.coeff_nm_tuple
        return f"Forbes Q-2D Coeff (n={n}, m={m}), Surface {self.surface_number}"
