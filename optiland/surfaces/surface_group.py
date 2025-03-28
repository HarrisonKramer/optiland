"""Surface Group

This module contains the SurfaceGroup class, which represents a group of
surfaces in an optical system. The SurfaceGroup class provides methods for
tracing rays through the surfaces, adding and removing surfaces, and
converting the group to and from a dictionary for serialization.

Kramer Harrison, 2024
"""

from copy import deepcopy

import numpy as np

from optiland.coatings import BaseCoatingPolarized
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.surface_factory import SurfaceFactory


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

    @property
    def x(self):
        """np.array: x intersection points on all surfaces"""
        return np.array([surf.x for surf in self.surfaces if surf.x.size > 0])

    @property
    def y(self):
        """np.array: y intersection points on all surfaces"""
        return np.array([surf.y for surf in self.surfaces if surf.y.size > 0])

    @property
    def z(self):
        """np.array: z intersection points on all surfaces"""
        return np.array([surf.z for surf in self.surfaces if surf.z.size > 0])

    @property
    def L(self):
        """np.array: x direction cosines on all surfaces"""
        return np.array([surf.L for surf in self.surfaces if surf.L.size > 0])

    @property
    def M(self):
        """np.array: y direction cosines on all surfaces"""
        return np.array([surf.M for surf in self.surfaces if surf.M.size > 0])

    @property
    def N(self):
        """np.array: z direction cosines on all surfaces"""
        return np.array([surf.N for surf in self.surfaces if surf.N.size > 0])

    @property
    def opd(self):
        """np.array: optical path difference recorded on all surfaces"""
        return np.array([surf.opd for surf in self.surfaces if surf.opd.size > 0])

    @property
    def u(self):
        """np.array: paraxial ray angles on all surfaces"""
        return np.array([surf.u for surf in self.surfaces if surf.u.size > 0])

    @property
    def intensity(self):
        """np.array: ray intensities on all surfaces"""
        return np.array(
            [surf.intensity for surf in self.surfaces if surf.intensity.size > 0],
        )

    @property
    def positions(self):
        """np.array: z positions of surface vertices"""
        return np.array([surf.geometry.cs.position_in_gcs[2] for surf in self.surfaces])

    @property
    def radii(self):
        """np.array: radii of curvature of all surfaces"""
        return np.array([surf.geometry.radius for surf in self.surfaces])

    @property
    def conic(self):
        """np.array: conic constant of all surfaces"""
        values = []
        for surf in self.surfaces:
            try:
                values.append(surf.geometry.k)
            except AttributeError:
                values.append(0)
        return np.array(values)

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

        self.surfaces.insert(index, new_surface)

        self.surface_factory.last_thickness = kwargs.get("thickness", 0)

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

    def inverted(self):
        """Generate inverted surface group.

        This method generates an inverted surface group by performing the
            following operations:
            1. Reverses the order of the surfaces in the original surface
                group.
            2. Scales the radii of each surface by -1.
            3. Inverts the z position of each surface by subtracting it from
                the z position of the last surface.
            4. Swaps the initial and final materials of each surface.

        Returns:
            SurfaceGroup: The inverted surface group.

        """
        surfs_inverted = deepcopy(self.surfaces[::-1])
        z_shift = self.surfaces[-1].geometry.cs.z
        for surf in surfs_inverted:
            # scale radii by -1
            surf.geometry.radius *= -1

            # invert z position
            surf.geometry.cs.z = z_shift - surf.geometry.cs.z

            # swap initial and final materials
            temp = surf.material_pre
            surf.material_pre = surf.material_post
            surf.material_post = temp

        return SurfaceGroup(surfs_inverted)

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
