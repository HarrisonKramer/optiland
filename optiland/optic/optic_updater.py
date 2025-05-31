"""Optic Updater Module

This module contains the OpticModifier class, which is responsible for updating the
optical system properties, such as the surface radii of curvature, thicknesses,
materials, conic constants, polarization, etc.

Kramer Harrison, 2024
"""
import warnings
from typing import Union

import optiland.backend as be
from optiland.geometries import Plane, StandardGeometry
from optiland.materials import IdealMaterial
from optiland.rays import PolarizationState


class OpticUpdater:
    """Class to update or modify an optical system

    This class is responsible for updating the optical system properties, such as
    the surface radii of curvature, thicknesses, materials, conic constants,
    polarization, etc.

    Args:
        optic (Optic): The optical system to be modified.
    """

    def __init__(self, optic):
        self.optic = optic

    def set_radius(self, value, surface_number):
        """Set the radius of curvature of a surface.

        Args:
            value (float): The new radius of curvature.
            surface_number (int): The index of the surface to modify.

        """
        surface = self.optic.surface_group.surfaces[surface_number]

        # change geometry from plane to standard
        if isinstance(surface.geometry, Plane):
            cs = surface.geometry.cs
            new_geometry = StandardGeometry(cs, radius=value, conic=0)
            surface.geometry = new_geometry
        else:
            surface.geometry.radius = value

    def set_conic(self, value, surface_number):
        """Set the conic constant of a surface.

        Args:
            value (float): The new conic constant.
            surface_number (int): The index of the surface to modify.

        """
        surface = self.optic.surface_group.surfaces[surface_number]
        surface.geometry.k = value

    def set_thickness(self, value, surface_number):
        """Set the thickness of a surface.

        Args:
            value (float): The new thickness value to set for the space
                following the specified surface.
            surface_number (int): The index of the surface *before* the
                thickness to be modified.

        """
        positions = self.optic.surface_group.positions
        delta_t = value - positions[surface_number + 1] + positions[surface_number]
        positions = be.copy(positions)  # required to avoid in-place modification
        positions[surface_number + 1 :] = positions[surface_number + 1 :] + delta_t
        positions = positions - positions[1]  # force surface 1 to be at zero
        for k, surface in enumerate(self.optic.surface_group.surfaces):
            surface.geometry.cs.z = be.array(positions[k])

    def set_index(self, value, surface_number):
        """Set the index of refraction of a surface.

        Args:
            value (float): The new refractive index value.
            surface_number (int): The index of the surface whose *post-material*
                (material after the surface) will be updated. This also updates
                the *pre-material* of the subsequent surface.

        """
        surface = self.optic.surface_group.surfaces[surface_number]
        new_material = IdealMaterial(n=value, k=0)
        surface.material_post = new_material

        surface_post = self.optic.surface_group.surfaces[surface_number + 1]
        surface_post.material_pre = new_material

    def set_asphere_coeff(self, value, surface_number, aspher_coeff_idx):
        """Set the asphere coefficient on a surface

        Args:
            value (float): The new value for the aspheric coefficient.
            surface_number (int): The index of the surface to modify.
            aspher_coeff_idx (int): The index of the aspheric coefficient
                within the surface's coefficient list to set.

        """
        surface = self.optic.surface_group.surfaces[surface_number]
        surface.geometry.c[aspher_coeff_idx] = value

    def set_polarization(self, polarization: Union[PolarizationState, str]):
        """Set the polarization state of the optic.

        Args:
            polarization (PolarizationState or str): The polarization
                state to set. Can be a `PolarizationState` object or the string
                'ignore'.

        """
        if isinstance(polarization, str) and polarization != "ignore":
            raise ValueError(
                "Invalid polarization state. Must be either "
                'PolarizationState or "ignore".',
            )
        self.optic.polarization = polarization

    def scale_system(self, scale_factor):
        """Scales the optical system by a given scale factor.

        Args:
            scale_factor (float): The factor by which to scale all relevant
                system dimensions (radii, thicknesses, EPD, physical apertures).

        """
        num_surfaces = self.optic.surface_group.num_surfaces
        radii = self.optic.surface_group.radii
        thicknesses = [
            self.optic.surface_group.get_thickness(surf_idx)[0]
            for surf_idx in range(num_surfaces - 1)
        ]

        # Scale radii & thicknesses
        for surf_idx in range(num_surfaces):
            if not be.isinf(radii[surf_idx]):
                self.set_radius(radii[surf_idx] * scale_factor, surf_idx)

            if surf_idx != num_surfaces - 1 and not be.isinf(thicknesses[surf_idx]):
                self.set_thickness(thicknesses[surf_idx] * scale_factor, surf_idx)

        # Scale aperture, if aperture type is EPD
        if self.optic.aperture.ap_type in ["EPD", "float_by_stop_size"]:
            self.optic.aperture.value = self.optic.aperture.value * scale_factor

        # Scale physical apertures
        for surface in self.optic.surface_group.surfaces:
            if surface.aperture is not None:
                surface.aperture.scale(scale_factor)

    def update_paraxial(self):
        """Update the semi-aperture of all surfaces based on paraxial marginal
        and chief ray heights. Also updates normalization radii for relevant
        surface types.
        """
        ya, _ = self.optic.paraxial.marginal_ray()
        yb, _ = self.optic.paraxial.chief_ray()
        ya = be.abs(be.ravel(ya))
        yb = be.abs(be.ravel(yb))
        for k, surface in enumerate(self.optic.surface_group.surfaces):
            surface.set_semi_aperture(r_max=ya[k] + yb[k])
            self.update_normalization(surface)

    def update_normalization(self, surface) -> None:
        """Update the normalization radius/factors of a given non-spherical surface.

        The normalization factors (`norm_x`, `norm_y`, or `norm_radius`) are
        typically set to 1.1 times the surface's current semi-aperture.

        Args:
            surface (Surface): The surface whose normalization factors are to be
                updated.
        """
        if surface.surface_type in [
            "even_asphere",
            "odd_asphere",
            "polynomial",
            "chebyshev",
        ]:
            surface.geometry.norm_x = surface.semi_aperture * 1.1
            surface.geometry.norm_y = surface.semi_aperture * 1.1
        if surface.surface_type == "zernike":
            surface.geometry.norm_radius = surface.semi_aperture * 1.1

    def update(self) -> None:
        """Update the optical system by applying all defined pickups and solves.
        If certain surface types requiring paraxial updates are present,
        `update_paraxial()` is also called.
        """
        self.optic.pickups.apply()
        self.optic.solves.apply()

        if any(
            surface.surface_type in ["chebyshev", "zernike"]
            for surface in self.optic.surface_group.surfaces
        ):
            self.update_paraxial()

    def image_solve(self):
        """Adjusts the position of the image surface (last surface) such that
        the paraxial marginal ray crosses the optical axis at this new location.
        This effectively sets the paraxial focus.
        """
        ya, ua = self.optic.paraxial.marginal_ray()
        offset = float(ya[-1, 0] / ua[-1, 0])
        self.optic.surface_group.surfaces[-1].geometry.cs.z = (
            self.optic.surface_group.surfaces[-1].geometry.cs.z - offset
        )

    def flip(self):
        """Flips the optical system, reversing the order of surfaces (excluding
        object and image planes), their geometries, and materials. Pickups and
        solves referencing surface indices are updated accordingly.
        The new first optical surface (originally the last) is placed at z=0.0.
        """
        if not self.optic.surface_group or self.optic.surface_group.num_surfaces < 3:
            warnings.warn("Optic flip requires at least 3 surfaces (obj, element, img). No action taken.")
            return

        # 1. Capture original global Z-coordinates
        # Ensure conversion to standard Python floats, be.to_numpy handles scalar arrays
        original_z_coords = [
            float(be.to_numpy(surf.geometry.cs.z)) for surf in self.optic.surface_group.surfaces
        ]

        # 2. Call SurfaceGroup.flip()
        # This uses default start_index=1, end_index=-1 (flips elements between obj/img)
        self.optic.surface_group.flip(original_vertex_gcs_z_coords=original_z_coords)

        # 3. Define remapping function for indices
        # This remap function assumes 0-based indexing for all surface lists
        # and that the object (idx 0) and image (idx N-1) surfaces themselves
        # are not part of the re-ordering by SurfaceGroup.flip's default args,
        # but their properties (like material_post for obj, material_pre for img)
        # might be affected by their new neighbors' individual flips.
        # The indices for pickups/solves usually refer to the full list.
        # If SurfaceGroup.flip reorders surfaces s_1, ..., s_{N-2}
        # into s'_{N-2}, ..., s'_1 in place, then an original index k (1 <= k <= N-2)
        # now corresponds to a new physical surface.
        # The surface originally at index `old_idx` is now at `new_idx_in_list`.
        # Example: [S0, S1, S2, S3, S4] (N=5)
        # Flipped part: S1, S2, S3 -> S3', S2', S1'
        # New list: [S0, S3', S2', S1', S4]
        # Original S1 (idx 1) is now S1' at new_idx_in_list = 3. (N-1-1)
        # Original S2 (idx 2) is now S2' at new_idx_in_list = 2. (N-1-2)
        # Original S3 (idx 3) is now S3' at new_idx_in_list = 1. (N-1-3)
        # This is for the elements *within* the flipped segment.
        # Object (0) and Image (N-1) are not re-indexed in terms of list position.

        num_surfaces = self.optic.surface_group.num_surfaces

        # Default segment flipped is from index 1 to num_surfaces-2 (inclusive)
        # Let start_flip_idx = 1, end_flip_idx = num_surfaces - 2
        # An old_idx in this range [start_flip_idx, end_flip_idx]
        # new_idx_in_list = start_flip_idx + (end_flip_idx - old_idx)
        # new_idx_in_list = 1 + (num_surfaces - 2 - old_idx)

        def remap_index_func(old_idx):
            if old_idx == 0 or old_idx == num_surfaces - 1: # Object or Image surface
                return old_idx
            # For surfaces within the flipped segment (default 1 to N-2)
            # old_idx is 1-based for user, but 0-based in list. Assume 0-based here.
            # segment is from index 1 to num_surfaces-2
            if 1 <= old_idx <= num_surfaces - 2:
                # Relative index within the flipped part: old_idx - 1
                # New relative index after reversal: (num_surfaces - 2 - 1) - (old_idx - 1)
                # New absolute index: (num_surfaces - 3 - old_idx + 1) + 1 = num_surfaces - 1 - old_idx
                return num_surfaces - 1 - old_idx
            return old_idx # Should not happen if indices are valid

        # 4. Handle Pickups
        if hasattr(self.optic, 'pickups') and self.optic.pickups and \
           hasattr(self.optic.pickups, 'pickups') and len(self.optic.pickups.pickups) > 0:
            if hasattr(self.optic.pickups, 'remap_surface_indices'):
                self.optic.pickups.remap_surface_indices(remap_index_func)
            else:
                warnings.warn(
                    "PickupManager does not have remap_surface_indices method. Pickups may be incorrect."
                )

        # 5. Handle Solves
        if hasattr(self.optic, 'solves') and self.optic.solves and \
           hasattr(self.optic.solves, 'solves') and len(self.optic.solves.solves) > 0:
            if hasattr(self.optic.solves, 'remap_surface_indices'):
                self.optic.solves.remap_surface_indices(remap_index_func)
            else:
                warnings.warn(
                    "SolveManager does not have remap_surface_indices method. Solves may be incorrect."
                )

        # 6. Call self.optic.update()
        if hasattr(self.optic, 'update') and callable(self.optic.update):
            self.optic.update()
        elif hasattr(self, 'update') and callable(self.update): # OpticUpdater.update()
             self.update()


        # 7. Print confirmation
        print("Optical system has been flipped.")
