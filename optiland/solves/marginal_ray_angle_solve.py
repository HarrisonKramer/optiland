from optiland.solves.curvature_solve_base import CurvatureSolveBase
# Assuming Optic and ParaxialData types are available or handled via annotations
# For example, from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from optiland.optic.optic import Optic

import optiland.backend as be # For potential logging or array operations

class MarginalRayAngleSolve(CurvatureSolveBase):
    """Adjusts surface curvature to achieve a target marginal ray angle.

    This solve modifies the curvature of a specified surface such that the
    paraxial marginal ray emerges from that surface with a defined angle (slope).
    """

    def __init__(self, optic, surface_idx: int, angle: float):
        """Initializes MarginalRayAngleSolve.

        Args:
            optic: The main optical system object.
            surface_idx: Integer index of the surface whose curvature is to be
                         modified. The angle is targeted for the ray emerging
                         from this surface.
            angle: Float, the target marginal ray slope after refraction at
                   surface_idx.
        """
        super().__init__(optic, surface_idx, angle)

    def _get_paraxial_data_at_surface(self):
        """Fetches paraxial marginal ray data and refractive indices.

        Retrieves the marginal ray's height (y_k) and slope before refraction (u_k)
        at the specified surface (self.surface_idx). It also gets the
        refractive indices of the media immediately preceding (n_k) and
        succeeding (n_prime_k) this surface, for the primary wavelength.

        Returns:
            tuple: (y_k, u_k, n_k, n_prime_k) for the marginal ray at
                   self.surface_idx.

        Raises:
            IndexError: If surface_idx is out of bounds for ray data or
                        material data.
            AttributeError: If paraxial data or surface group is not available
                            in the optic object.
        """
        if self.optic.paraxial is None:
            raise AttributeError("Paraxial analysis data is not available in the optic system.")
        if self.optic.surface_group is None or not self.optic.surface_group.surfaces:
            raise AttributeError("Surface group is not available or is empty in the optic system.")

        # Fetch paraxial marginal ray data (heights y and slopes u)
        # y_values are at the surfaces, u_values are before refraction at surfaces
        y_values, u_values = self.optic.paraxial.marginal_ray()

        if not (0 <= self.surface_idx < len(y_values) and 0 <= self.surface_idx < len(u_values)):
            raise IndexError(
                f"MarginalRayAngleSolve: surface_idx {self.surface_idx} is out of bounds "
                f"for marginal ray data arrays (length y: {len(y_values)}, u: {len(u_values)})."
            )

        y_k = y_values[self.surface_idx]
        u_k = u_values[self.surface_idx] # This is u_k (slope *before* refraction)

        # Obtain refractive indices n_k and n_prime_k for self.surface_idx
        # This assumes a method like get_refractive_indices_at_surface(idx, wavelength_idx)
        # or direct access through surface material properties.
        # Let's assume self.optic.paraxial can provide this, or self.optic.surface_group.
        # Based on typical optical design software structure, often paraxial analysis
        # itself might store or have access to the indices used in its calculations.

        # Option 1: Paraxial class has a method (preferred if it exists)
        # n_k, n_prime_k = self.optic.paraxial.get_material_indices_at_surface(
        #    self.surface_idx, self.optic.get_primary_wavelength_idx() # Assuming such a method
        # )

        # Option 2: Access through surface_group and materials, then get index for primary wavelength
        # This is more complex as it requires knowing how materials are linked and evaluated.
        # For example:
        if not (0 <= self.surface_idx < len(self.optic.surface_group.surfaces)):
             raise IndexError(
                f"MarginalRayAngleSolve: surface_idx {self.surface_idx} is out of bounds "
                f"for surface group (length {len(self.optic.surface_group.surfaces)})."
            )

        # n_k: index of medium *before* surface_idx
        if self.surface_idx == 0:
            # For the first surface, n_k is the index of the object space medium.
            # This might be stored in optic.object_space_medium.index_at_wavelength(primary_wl)
            # or typically defaults to 1.0 (air) if not specified.
            # Let's assume paraxial setup handles this, or there's a convention.
            # A common approach is that paraxial.marginal_ray() itself is computed using these indices.
            # If self.optic.paraxial.indices exists and stores n and n' for each surface:
            if hasattr(self.optic.paraxial, 'indices_n') and \
               hasattr(self.optic.paraxial, 'indices_n_prime'):
                if not (0 <= self.surface_idx < len(self.optic.paraxial.indices_n) and \
                        0 <= self.surface_idx < len(self.optic.paraxial.indices_n_prime)):
                    raise IndexError(
                        f"MarginalRayAngleSolve: surface_idx {self.surface_idx} is out of bounds "
                        f"for paraxial refractive index arrays."
                    )
                n_k = self.optic.paraxial.indices_n[self.surface_idx]
                n_prime_k = self.optic.paraxial.indices_n_prime[self.surface_idx]
            else:
                # Fallback: try to get from surface materials directly for primary wavelength
                # This part is highly dependent on Optic's structure.
                # This is a placeholder for actual refractive index fetching logic.
                # The user prompt mentioned: "You'll need to determine the best way to get these"
                # This implies it might not be straightforward.
                # A robust system usually has paraxial.get_indices_for_surface(idx)
                # For now, let's assume a conventional way or raise NotImplementedError
                # if the structure isn't clear.
                # Based on typical structure, the paraxial ray trace itself would have used/stored these.
                # Let's try to access them from the surfaces, assuming primary wavelength.
                primary_wl = self.optic.wavelengths.primary_wavelength_value # Or similar

                current_surface = self.optic.surface_group.surfaces[self.surface_idx]

                if self.surface_idx == 0:
                    # Object space refractive index
                    # Assuming object surface material is None or special, and object space index is handled
                    # by paraxial setup or a default (e.g. 1.0 for air if optic.object_medium is None)
                    object_material = self.optic.surface_group.surfaces[0].material_before
                    if object_material:
                        n_k = object_material.get_index(primary_wl)
                    elif self.optic.object_space_material: # A common pattern
                        n_k = self.optic.object_space_material.get_index(primary_wl)
                    else: # Default to vacuum/air
                        n_k = 1.0
                else:
                    prev_surface = self.optic.surface_group.surfaces[self.surface_idx - 1]
                    if prev_surface.material_after:
                        n_k = prev_surface.material_after.get_index(primary_wl)
                    else:
                        # This case should ideally not happen in a well-defined system
                        # or means it's a mirror in air etc.
                        # Fallback or error needed here. For now, assume it's defined.
                        raise ValueError(f"Material after surface {self.surface_idx-1} is not defined.")

                if current_surface.material_after:
                    n_prime_k = current_surface.material_after.get_index(primary_wl)
                else:
                    # If it's the last surface before image, n_prime_k is image space index
                    # This logic can get complex.
                    # Safest bet: Paraxial system should provide these directly.
                    # If indices_n and indices_n_prime are not in self.optic.paraxial, this is an issue.
                    raise NotImplementedError(
                        "Refractive index determination logic needs to be confirmed. "
                        "Ideally, self.optic.paraxial.indices_n and "
                        "self.optic.paraxial.indices_n_prime should be available."
                    )

        # Ensure n_k and n_prime_k are floats, not None or other types
        if not isinstance(n_k, (float, int)) or not isinstance(n_prime_k, (float, int)):
            raise TypeError(
                f"Refractive indices n_k ({n_k}) and n_prime_k ({n_prime_k}) must be numeric."
            )

        return float(y_k), float(u_k), float(n_k), float(n_prime_k)

    @classmethod
    def from_dict(cls, optic, data):
        """Creates a MarginalRayAngleSolve instance from a dictionary.

        Args:
            optic: The optic object.
            data (dict): Dictionary containing 'surface_idx' and 'angle'.

        Returns:
            MarginalRayAngleSolve: An instance of this class.
        """
        if "surface_idx" not in data or "angle" not in data:
            raise ValueError(
                "Data for MarginalRayAngleSolve must include 'surface_idx' and 'angle'."
            )
        return cls(optic, data["surface_idx"], data["angle"])

```
