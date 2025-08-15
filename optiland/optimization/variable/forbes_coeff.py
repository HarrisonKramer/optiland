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
        self.coeff_number = coeff_number  # This is the radial order 'n'

    def get_value(self):
        """Gets the value of the nth Q-bfs coefficient from the radial_terms dict."""
        surf = self._surfaces.surfaces[self.surface_number]
        geom = surf.geometry
        if not isinstance(geom, ForbesQbfsGeometry):
            raise TypeError("This variable is only for ForbesQbfsGeometry.")

        # Access the public radial_terms dictionary directly
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

        # Update the public dictionary
        geom.radial_terms[self.coeff_number] = new_value

        # Trigger the geometry object to re-process the dictionary into its
        # internal coefficient lists for the computational backend.
        geom._prepare_coeffs()

    def scale(self, value):
        """
        So far, scaling is a no-op for Forbes coefficients.
        NOTE
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
    (n, m, type) term defined in the `freeform_coeffs` dictionary.
    """

    def __init__(
        self, optic, surface_number, coeff_tuple, apply_scaling=False, **kwargs
    ):
        super().__init__(optic, surface_number, apply_scaling, **kwargs)
        if not isinstance(coeff_tuple, tuple) or not (2 <= len(coeff_tuple) <= 3):
            raise ValueError(
                "`coeff_tuple` must be a tuple of (n, m) or (n, m, 'sin')."
            )
        self.coeff_tuple = coeff_tuple

    def get_value(self):
        """Gets the value of the coefficient for the specified (n, m, type) term."""
        surf = self._surfaces.surfaces[self.surface_number]
        geom = surf.geometry
        if not isinstance(geom, ForbesQ2dGeometry):
            raise TypeError("This variable is only for ForbesQ2dGeometry.")

        return geom.freeform_coeffs.get(self.coeff_tuple, 0.0)

    def update_value(self, new_value):
        """Updates the value of the coefficient for the specified (n, m, type) term."""
        surf = self.optic.surface_group.surfaces[self.surface_number]
        geom = surf.geometry

        geom.freeform_coeffs[self.coeff_tuple] = new_value
        geom._prepare_coeffs()

    def __str__(self):
        n, m, *tail = self.coeff_tuple
        term_type = "sin" if tail and tail[0] == "sin" else "cos"
        return (
            f"Forbes Q-2D Coeff (n={n}, m={m}, {term_type}), "
            f"Surface {self.surface_number}"
        )
