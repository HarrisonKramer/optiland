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
from optiland.surfaces.object_surface import ObjectSurface


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
        surface_to_add = None

        if new_surface is not None:
            surface_to_add = new_surface
            # Apply explicit parameters from add_surface call to the provided surface object
            surface_to_add.is_stop = is_stop
            surface_to_add.comment = comment # Overwrite comment if provided in args
        else:
            if index is None:
                 raise ValueError("Must define index when creating a new surface from type (when 'new_surface' is not provided).")

            surface_to_add = self.surface_factory.create_surface(
                surface_type=surface_type,
                comment=comment,
                index=index,
                is_stop=is_stop,
                material=material,
                **kwargs,
            )

        if surface_to_add.is_stop:
            for srf in self.surfaces:
                srf.is_stop = False

        # *** NEW VALIDATION CHECKS START HERE ***
        if index is not None:
            if not isinstance(index, int):
                raise TypeError(f"Surface index must be an integer, got {type(index).__name__}.")
            if index < 0:
                raise ValueError(f"Surface index cannot be negative, got {index}.")
            # Allow index == len(self.surfaces) for inserting at the very end.
            # Disallow index > len(self.surfaces) as it would create a gap.
            if index > len(self.surfaces):
                raise ValueError(
                    f"Surface index {index} is out of bounds. With {len(self.surfaces)} surfaces, "
                    f"the maximum allowed index for insertion is {len(self.surfaces)} (to append). "
                    f"Cannot create gaps."
                )
        # *** NEW VALIDATION CHECKS END HERE ***

        if index == 0:
            if not isinstance(surface_to_add, ObjectSurface):
                raise ValueError(
                    f"Surface at index 0 must be an ObjectSurface, got {type(surface_to_add).__name__}."
                )

            if self.surfaces:
                self.surfaces[0] = surface_to_add
            else:
                self.surfaces.insert(0, surface_to_add)
            return

        if index is None:
            self.surfaces.append(surface_to_add)
        else:
            self.surfaces.insert(index, surface_to_add)

    def remove_surface(self, index):
        """Remove a surface from the list of surfaces.

        Args:
            index (int): The index of the surface to remove.

        Raises:
            ValueError: If the index is 0 (object surface).

        Returns:
        None

        """
        if index == 0:
            raise ValueError("Cannot remove object surface.")
        del self.surfaces[index]

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
