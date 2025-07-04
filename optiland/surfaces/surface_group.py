"""Surface Group

This module contains the SurfaceGroup class, which represents a group of
surfaces in an optical system. The SurfaceGroup class provides methods for
tracing rays through the surfaces, adding and removing surfaces, and
converting the group to and from a dictionary for serialization.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.coatings import BaseCoatingPolarized
from optiland.surfaces.factories.surface_factory import SurfaceFactory
from optiland.surfaces.standard_surface import Surface


class SurfaceGroup:
    """Represents a group of surfaces in an optical system.

    Attributes:
        surfaces (list): List of surfaces in the group.
        _last_thickness (float): The thickness of the last surface added.

    """

    def __init__(self, surfaces: list[Surface] = None):
        """Initializes a new instance of the SurfaceGroup class.

        Args:
            surfaces (List, optional): List of surfaces to initialize the
                group with. Defaults to None.

        """
        if surfaces is None:
            self.surfaces = []
        else:
            self.surfaces = surfaces

        self.surface_factory = SurfaceFactory(self)

    def __add__(self, other):
        """Add two SurfaceGroup objects together.

        Note that this ignores the image surface of the current group and the object
        surface of the other group.
        """
        # add the offset of the last surface in self to each surface in other
        offset = self.surfaces[-1].geometry.cs.z

        # add object surface distance if finite
        object_distance = other.surfaces[0].geometry.cs.z
        if be.isfinite(object_distance):
            offset = offset - object_distance

        for surf in other.surfaces[1:]:
            surf.geometry.cs.z = surf.geometry.cs.z + offset

        # remove stop surface from other
        for surface in other.surfaces:
            surface.is_stop = False

        return SurfaceGroup(self.surfaces[:-1] + other.surfaces[1:])

    @property
    def x(self):
        """np.array: x intersection points on all surfaces"""
        return be.stack([surf.x for surf in self.surfaces if be.size(surf.x) > 0])

    @property
    def y(self):
        """np.array: y intersection points on all surfaces"""
        return be.stack([surf.y for surf in self.surfaces if be.size(surf.y) > 0])

    @property
    def z(self):
        """np.array: z intersection points on all surfaces"""
        return be.stack([surf.z for surf in self.surfaces if be.size(surf.z) > 0])

    @property
    def L(self):
        """np.array: x direction cosines on all surfaces"""
        return be.stack([surf.L for surf in self.surfaces if be.size(surf.L) > 0])

    @property
    def M(self):
        """np.array: y direction cosines on all surfaces"""
        return be.stack([surf.M for surf in self.surfaces if be.size(surf.M) > 0])

    @property
    def N(self):
        """np.array: z direction cosines on all surfaces"""
        return be.stack([surf.N for surf in self.surfaces if be.size(surf.N) > 0])

    @property
    def opd(self):
        """np.array: optical path difference recorded on all surfaces"""
        return be.stack([surf.opd for surf in self.surfaces if be.size(surf.opd) > 0])

    @property
    def u(self):
        """np.array: paraxial ray angles on all surfaces"""
        return be.stack([surf.u for surf in self.surfaces if be.size(surf.u) > 0])

    @property
    def intensity(self):
        """np.array: ray intensities on all surfaces"""
        return be.stack(
            [surf.intensity for surf in self.surfaces if be.size(surf.intensity) > 0]
        )

    @property
    def positions(self):
        """np.array: z positions of surface vertices"""
        positions = be.array(
            [surf.geometry.cs.position_in_gcs[2] for surf in self.surfaces]
        )
        return positions.reshape(-1, 1)

    @property
    def radii(self):
        """np.array: radii of curvature of all surfaces"""
        return be.array([surf.geometry.radius for surf in self.surfaces])

    @property
    def conic(self):
        """be.array: conic constant of all surfaces"""
        values = []
        for surf in self.surfaces:
            try:
                values.append(surf.geometry.k)
            except AttributeError:
                values.append(0)
        return be.array(values)

    @property
    def stop_index(self):
        """int: the index of the aperture stop surface"""
        for index, surface in enumerate(self.surfaces):
            if surface.is_stop:
                return index

        raise ValueError("No stop surface found.")

    @property
    def num_surfaces(self):
        """int: the number of surfaces"""
        return len(self.surfaces)

    @property
    def uses_polarization(self):
        """bool: True if any surface uses polarization, False otherwise"""
        for surf in self.surfaces:
            if isinstance(surf.coating, BaseCoatingPolarized):
                return True
        return False

    @property
    def total_track(self):
        """float: the total track length of the system"""
        if self.num_surfaces < 2:
            raise ValueError("Not enough surfaces to calculate total track.")
        z = self.positions[1:-1]
        return be.max(z) - be.min(z)

    def n(self, wavelength):
        """Get the refractive indices of the surfaces.

        Args:
            wavelength (float or str, optional): The wavelength for which to
                calculate the refractive indices.

        Returns:
            numpy.ndarray: The refractive indices of the surfaces.

        """
        n = []
        for surface in self.surfaces:
            n.append(be.atleast_1d(surface.material_post.n(wavelength)))
        return be.ravel(be.array(n))

    def get_thickness(self, surface_number):
        """Calculate the thickness between two surfaces.

        Args:
            surface_number (int): The index of the first surface.

        Returns:
            float: The thickness between the two surfaces.

        """
        t = self.positions
        return t[surface_number + 1] - t[surface_number]

    def trace(self, rays, skip=0):
        """Trace the given rays through the surfaces.

        Args:
            rays (BaseRays): List of rays to be traced.
            skip (int, optional): Number of surfaces to skip before tracing.
                Defaults to 0.

        """
        self.reset()
        for surface in self.surfaces[skip:]:
            surface.trace(rays)
        return rays

    def add_surface(
        self,
        new_surface=None,
        surface_type="standard",
        comment="",
        index=None,
        is_stop=False,
        material="air",
        **kwargs,
    ):
        """Adds a new surface to the list of surfaces.

        Args:
            new_surface (Surface, optional): The new surface to add. If not
                provided, a new surface will be created based on the other
                arguments.
            surface_type (str, optional): The type of surface to create.
            comment (str, optional): A comment for the surface. Defaults to ''.
            index (int, optional): The index at which to insert the new
                surface. If not provided, the surface will be appended to the
                end of the list.
            is_stop (bool, optional): Indicates if the surface is the aperture.
            material (str, optional): The material of the surface.
                Default is 'air'.
            **kwargs: Additional keyword arguments for surface-specific
                parameters such as radius, conic, dx, dy, rx, ry, rz, aperture,
                bsdf, x, y, z.

        Raises:
            ValueError: If index is not provided when defining a new surface.

        """
        if new_surface is None:
            if index is None:
                raise ValueError("Must define index when defining surface.")

            new_surface = self.surface_factory.create_surface(
                surface_type,
                comment,
                index,
                is_stop,
                material,
                **kwargs,
            )

        if new_surface.is_stop:
            for surface in self.surfaces:
                surface.is_stop = False

        if index is None:
            self.surfaces.append(new_surface)
        else:
            if index < 0:
                raise IndexError(f"Index {index} cannot be negative.")
            if index > len(self.surfaces):
                raise IndexError(
                    f"Index {index} is out of bounds for insertion. "
                    f"Max index for insertion is {len(self.surfaces)} (to append)."
                )

            self.surfaces.insert(index, new_surface)

            # If a surface was inserted (not appended) and there's a surface after it
            if index < len(self.surfaces) - 1:
                surface_after_inserted = self.surfaces[index + 1]
                new_surface = self.surfaces[index]
                surface_after_inserted.material_pre = new_surface.material_post

            # Update coordinate systems if surface was inserted
            if not self.surface_factory.use_absolute_cs and index < (
                len(self.surfaces) - 1
            ):
                self._update_coordinate_systems(start_index=index)

    def remove_surface(self, index):
        """Remove a surface from the list of surfaces.

        Cannot remove the object surface (index 0).
        If relative coordinate positioning is active (use_absolute_cs=False),
        this may trigger an update of subsequent surface positions.

        Args:
            index (int): The index of the surface to remove.

        Raises:
            ValueError: If attempting to remove the object surface (index 0).
            IndexError: If the index is out of bounds for the current list of surfaces.
        """
        if index == 0:
            raise ValueError("Cannot remove object surface (index 0).")

        if not (0 < index < len(self.surfaces)):
            raise IndexError(
                f"Index {index} is out of bounds for removing from list of "
                f"{len(self.surfaces)} surfaces."
            )

        num_surfaces_before_removal = len(self.surfaces)

        del self.surfaces[index]

        if not self.surface_factory.use_absolute_cs:
            was_not_last_surface = index < num_surfaces_before_removal - 1
            if was_not_last_surface:
                self._update_coordinate_systems(start_index=index)

    def reset(self):
        """Resets all the surfaces in the collection.

        This method iterates over each surface in the collection and calls
            its `reset` method.
        """
        for surface in self.surfaces:
            surface.reset()

    def set_fresnel_coatings(self):
        """Set Fresnel coatings on all surfaces in the group."""
        for surface in self.surfaces[1:-1]:
            if surface.material_pre != surface.material_post:
                surface.set_fresnel_coating()

    def to_dict(self):
        """Convert the surface group to a dictionary.

        Returns:
            dict: The surface group as a dictionary.

        """
        return {"surfaces": [surface.to_dict() for surface in self.surfaces]}

    @classmethod
    def from_dict(cls, data):
        """Create a surface group from a dictionary.

        Args:
            data (dict): The dictionary to create the surface group from.

        Returns:
            SurfaceGroup: The surface group created from the dictionary.

        """
        return cls(
            [Surface.from_dict(surface_data) for surface_data in data["surfaces"]],
        )

    def _update_coordinate_systems(self, start_index):
        """Updates the coordinate systems of surfaces from start_index.

        This method is called when a surface is added, removed, or modified
        in a way that might affect the positions of subsequent surfaces,
        but only if absolute coordinate positioning (use_absolute_cs=True)
        is not being used by the coordinate system factory.

        It recalculates the z-coordinate of each surface based on the
        z-coordinate and 'thickness' attribute of the preceding surface.

        Args:
            start_index (int): The index of the surface from which to start
                            updating coordinate systems. The surface at
                            `start_index` itself will be updated if it's
                            not the object surface (index 0) and has a predecessor.
                            If `start_index` is 0, updates effectively begin
                            for surface 1 based on surface 0.
        """
        if not self.surfaces:
            return

        effective_start_index = start_index
        if start_index == 0:
            effective_start_index = 1

        if effective_start_index >= len(self.surfaces):
            return

        for i in range(effective_start_index, len(self.surfaces)):
            current_surface = self.surfaces[i]

            if i == 0:  # no update to object surface
                continue
            elif i == 1:  # first surface lies at z=0.0 by definition
                new_z = 0.0
            else:
                prev_surface = self.surfaces[i - 1]
                thickness = prev_surface.thickness

                if hasattr(thickness, "item"):
                    thickness = thickness.item()

                if be.isinf(thickness):
                    raise ValueError(
                        f"Coordinate system update failed due to infinite "
                        f"thickness at surface {start_index - 1}"
                    )

                new_z = prev_surface.geometry.cs.z + thickness

            current_surface.geometry.cs.z = be.array(new_z)

    def flip(
        self,
        original_vertex_gcs_z_coords: list[float],
        start_index: int = 1,
        end_index: int = -1,
    ):
        """Flips a segment of the surfaces in the group.

        Args:
            original_vertex_gcs_z_coords (list[float]): List of the original
                global Z-coordinates of all surface vertices in the group
                before flipping.
            start_index (int, optional): The starting index of the segment of
                surfaces to flip. Defaults to 1 (skips object surface).
            end_index (int, optional): The ending index (exclusive for positive,
                inclusive for negative slice behavior) of the segment of surfaces
                to flip. Defaults to -1 (up to, but not including, the image surface).
        """
        n_surfaces_total = len(self.surfaces)

        if start_index < 0:
            start_index = n_surfaces_total + start_index

        if end_index < 0:
            actual_slice_end_index = (
                n_surfaces_total + end_index if end_index != 0 else 0
            )
        else:
            actual_slice_end_index = end_index

        if start_index >= actual_slice_end_index:
            # No surfaces to flip or invalid range
            self.reset()
            return

        original_indices_in_segment = list(range(start_index, actual_slice_end_index))

        if not original_indices_in_segment:
            self.reset()
            return

        # Extract the segment, reverse it, and place it back
        segment_to_reverse = self.surfaces[start_index:actual_slice_end_index]
        segment_to_reverse.reverse()
        self.surfaces[start_index:actual_slice_end_index] = segment_to_reverse

        # Call flip() on each surface within the now-reversed segment
        for i in range(len(segment_to_reverse)):
            surface_index_in_group = start_index + i
            self.surfaces[surface_index_in_group].flip()

        if segment_to_reverse:  # Check if the segment is not empty
            self.surfaces[start_index].geometry.cs.z = 0.0

            # Iterate for thicknesses within the segment
            # k iterates from 0 to len(segment_to_reverse) - 2
            for k in range(len(segment_to_reverse) - 1):
                current_surf_in_new_order = self.surfaces[start_index + k]
                next_surf_in_new_order = self.surfaces[start_index + k + 1]

                original_idx_of_new_k = original_indices_in_segment[
                    len(segment_to_reverse) - 1 - k
                ]
                original_idx_of_new_k_plus_1 = original_indices_in_segment[
                    len(segment_to_reverse) - 1 - (k + 1)
                ]

                # The thickness is between these two original surfaces
                thickness = abs(
                    original_vertex_gcs_z_coords[original_idx_of_new_k]
                    - original_vertex_gcs_z_coords[original_idx_of_new_k_plus_1]
                )

                next_surf_in_new_order.geometry.cs.z = (
                    current_surf_in_new_order.geometry.cs.z + thickness
                )

        self.reset()
