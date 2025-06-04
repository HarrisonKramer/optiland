from optiland.solves.curvature_solve_base import CurvatureSolveBase
import optiland.backend as be


class ChiefRayAngleSolve(CurvatureSolveBase):
    """Adjusts surface curvature to achieve a target chief ray angle.

    This solve modifies the curvature of a specified surface such that the
    paraxial chief ray emerges from that surface with a defined angle (slope).
    """

    def __init__(self, optic, surface_idx: int, angle: float):
        """Initializes ChiefRayAngleSolve.

        Args:
            optic: The main optical system object.
            surface_idx: Integer index of the surface whose curvature is to be
                         modified. The angle is targeted for the ray emerging
                         from this surface.
            angle: Float, the target chief ray slope after refraction at
                   surface_idx.
        """
        super().__init__(optic, surface_idx, angle)

    def _get_paraxial_data_at_surface(self):
        """Fetches paraxial chief ray data and refractive indices.

        Retrieves the chief ray's height (y_bar_k) and slope before refraction
        (u_bar_k) at the specified surface (self.surface_idx). It also gets the
        refractive indices of the media immediately preceding (n_k) and
        succeeding (n_prime_k) this surface, for the primary wavelength.

        Returns:
            tuple: (y_bar_k, u_bar_k, n_k, n_prime_k) for the chief ray at
                   self.surface_idx.

        Raises:
            IndexError: If surface_idx is out of bounds for ray data or
                        material data.
            AttributeError: If paraxial data or surface group is not available
                            in the optic object.
        """
        if self.optic.paraxial is None:
            raise AttributeError(
                "Paraxial analysis data is not available in the optic system."
            )
        if self.optic.surface_group is None or not self.optic.surface_group.surfaces:
            raise AttributeError(
                "Surface group is not available or is empty in the optic system."
            )

        # Fetch paraxial chief ray data (heights y_bar and slopes u_bar)
        # y_bar_values are at the surfaces, u_bar_values are before refraction at surfaces
        y_bar_values, u_bar_values = self.optic.paraxial.chief_ray()

        if not (
            0 <= self.surface_idx < len(y_bar_values)
            and 0 <= self.surface_idx < len(u_bar_values)
        ):
            raise IndexError(
                f"ChiefRayAngleSolve: surface_idx {self.surface_idx} is out of bounds "
                f"for chief ray data arrays (length y_bar: {len(y_bar_values)}, u_bar: {len(u_bar_values)})."
            )

        y_bar_k = y_bar_values[self.surface_idx]
        u_bar_k = u_bar_values[
            self.surface_idx
        ]  # This is u_bar_k (slope *before* refraction)

        # Obtain refractive indices n_k and n_prime_k
        # Assuming the same logic as in MarginalRayAngleSolve for fetching indices,
        # i.e., relying on self.optic.paraxial.indices_n and self.optic.paraxial.indices_n_prime
        if hasattr(self.optic.paraxial, "indices_n") and hasattr(
            self.optic.paraxial, "indices_n_prime"
        ):
            if not (
                0 <= self.surface_idx < len(self.optic.paraxial.indices_n)
                and 0 <= self.surface_idx < len(self.optic.paraxial.indices_n_prime)
            ):
                raise IndexError(
                    f"ChiefRayAngleSolve: surface_idx {self.surface_idx} is out of bounds "
                    f"for paraxial refractive index arrays."
                )
            n_k = self.optic.paraxial.indices_n[self.surface_idx]
            n_prime_k = self.optic.paraxial.indices_n_prime[self.surface_idx]
        else:
            # Fallback logic (same as MarginalRayAngleSolve, highly dependent on Optic structure)
            # This path is less ideal and more prone to needing adjustment.
            primary_wl = self.optic.wavelengths.primary_wavelength_value
            if not (0 <= self.surface_idx < len(self.optic.surface_group.surfaces)):
                raise IndexError(
                    f"ChiefRayAngleSolve: surface_idx {self.surface_idx} is out of bounds "
                    f"for surface group (length {len(self.optic.surface_group.surfaces)})."
                )
            current_surface = self.optic.surface_group.surfaces[self.surface_idx]
            if self.surface_idx == 0:
                object_material = self.optic.surface_group.surfaces[0].material_before
                if object_material:
                    n_k = object_material.get_index(primary_wl)
                elif self.optic.object_space_material:
                    n_k = self.optic.object_space_material.get_index(primary_wl)
                else:
                    n_k = 1.0
            else:
                prev_surface = self.optic.surface_group.surfaces[self.surface_idx - 1]
                if prev_surface.material_after:
                    n_k = prev_surface.material_after.get_index(primary_wl)
                else:
                    raise ValueError(
                        f"Material after surface {self.surface_idx - 1} is not defined for chief ray."
                    )

            if current_surface.material_after:
                n_prime_k = current_surface.material_after.get_index(primary_wl)
            else:
                raise NotImplementedError(
                    "Refractive index determination logic for chief ray needs to be confirmed, "
                    "especially if self.optic.paraxial.indices_n/n_prime are not available."
                )

        if not isinstance(n_k, (float, int)) or not isinstance(n_prime_k, (float, int)):
            raise TypeError(
                f"Refractive indices n_k ({n_k}) and n_prime_k ({n_prime_k}) must be numeric for chief ray."
            )

        return float(y_bar_k), float(u_bar_k), float(n_k), float(n_prime_k)

    @classmethod
    def from_dict(cls, optic, data):
        """Creates a ChiefRayAngleSolve instance from a dictionary.

        Args:
            optic: The optic object.
            data (dict): Dictionary containing 'surface_idx' and 'angle'.

        Returns:
            ChiefRayAngleSolve: An instance of this class.
        """
        if "surface_idx" not in data or "angle" not in data:
            raise ValueError(
                "Data for ChiefRayAngleSolve must include 'surface_idx' and 'angle'."
            )
        return cls(optic, data["surface_idx"], data["angle"])
