"""This module provides specialized variable handlers for Forbes polynomial
coefficients, distinguishing between the rotationally symmetric Q-bfs
surfaces and the freeform Q-2D surfaces.

Manuel Fragata Mendes, August 2025
"""

from __future__ import annotations

from optiland.geometries.forbes.geometry import ForbesQ2dGeometry, ForbesQbfsGeometry
from optiland.optimization.variable.base import VariableBehavior


class ForbesQbfsCoeffVariable(VariableBehavior):
    """
    Represents a variable for a Forbes Q-bfs (rotationally symmetric) coefficient.
    This variable targets a coefficient for a specific radial order `n`.
    """

    def __init__(
        self, optic, surface_number, coeff_number, apply_scaling=False, **kwargs
    ):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)
        self.coeff_number = coeff_number  # the radial order 'n'

    def get_value(self):
        """Gets the value of the nth Q-bfs coefficient from the radial_terms dict."""
        surf = self._surfaces.surfaces[self.surface_number]
        geom = surf.geometry
        if not isinstance(geom, ForbesQbfsGeometry):
            raise TypeError("This variable is only for ForbesQbfsGeometry.")

        value = geom.radial_terms.get(self.coeff_number, 0.0)

        if self.apply_scaling:
            return self.scale(value)
        return value

    def update_value(self, new_value):
        """Updates the value of the nth Q-bfs coefficient in the radial_terms dict."""
        if self.apply_scaling:
            new_value = self.inverse_scale(new_value)

        surf = self.optic.surface_group.surfaces[self.surface_number]
        geom = surf.geometry

        geom.radial_terms[self.coeff_number] = new_value
        geom._prepare_coeffs()

    def scale(self, value):
        scaling_factor = 1.0
        return value * scaling_factor

    def inverse_scale(self, scaled_value):
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
    (type, m, n) term, following the Zemax convention.
    """

    def __init__(
        self, optic, surface_number, coeff_tuple, apply_scaling=False, **kwargs
    ):
        """
        Initializes the ForbesQ2dCoeffVariable.

        Parameters
        ----------
        optic : Optic
            The optical system.
        surface_number : int
            The surface number to which the coefficient belongs.
        coeff_tuple : tuple
            The identifier for the coefficient, following the Zemax convention:
            - ('a', m, n) for a cosine term a_n^m
            - ('b', m, n) for a sine term b_n^m
        """
        super().__init__(optic, surface_number, apply_scaling, **kwargs)
        self.attribute = "freeform_coeffs"

        # Parse the Zemax-style key for correct string representation and validation
        try:
            term_char, m, n = coeff_tuple
            self.m = m
            self.n = n
            if term_char.lower() == "a":
                self.term_type = "cos"
            elif term_char.lower() == "b":
                self.term_type = "sin"
            else:
                raise ValueError("Term type in coeff_tuple must be 'a' or 'b'.")
        except (ValueError, TypeError, IndexError) as err:
            raise ValueError(
                "coeff_tuple for ForbesQ2dCoeffVariable must be a tuple of the "
                "form ('a', m, n) or ('b', m, n)."
            ) from err

        self.coeff_tuple = coeff_tuple

    def get_value(self):
        """Gets the value of the coefficient for the specified term."""
        surf = self._surfaces.surfaces[self.surface_number]
        geom = surf.geometry
        if not isinstance(geom, ForbesQ2dGeometry):
            raise TypeError("This variable is only for ForbesQ2dGeometry.")

        return geom.freeform_coeffs.get(self.coeff_tuple, 0.0)

    def update_value(self, new_value):
        """Updates the value of the coefficient for the specified term."""
        surf = self.optic.surface_group.surfaces[self.surface_number]
        geom = surf.geometry

        geom.freeform_coeffs[self.coeff_tuple] = new_value
        geom._prepare_coeffs()

    def __str__(self):
        """Returns a string representation of the variable."""
        return (
            f"Forbes Q-2D Coeff (n={self.n}, m={self.m}, {self.term_type}), "
            f"Surface {self.surface_number}"
        )
