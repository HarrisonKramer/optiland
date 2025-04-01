"""Surface Factory

This module contains the SurfaceFactory class, which is used to create surface
objects based on the given parameters. The SurfaceFactory class is used by the
SurfaceGroup class to create surfaces for the optical system. The class
abstracts the creation of surface objects and allows for easy configuration of
the surface parameters.

Kramer Harrison, 2024
"""

from optiland.surfaces.factories.coating_factory import CoatingFactory
from optiland.surfaces.factories.coordinate_system_factory import (
    CoordinateSystemFactory,
)
from optiland.surfaces.factories.geometry_factory import GeometryFactory
from optiland.surfaces.factories.material_factory import MaterialFactory
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.paraxial_surface import ParaxialSurface
from optiland.surfaces.standard_surface import Surface


class SurfaceFactory:
    """A factory class for creating surface objects by delegating to sub-factories.

    Args:
        surface_group (SurfaceGroup): The surface group to which the surfaces belong.

    Attributes:
        _surface_group (SurfaceGroup): The surface group to which the surfaces belong.
        _coordinate_factory (CoordinateSystemFactory): Factory for coordinate systems.
        _geometry_factory (GeometryFactory): Factory for surface geometries.
        _material_factory (MaterialFactory): Factory for materials.
        _coating_factory (CoatingFactory): Factory for coatings.
    """

    def __init__(self, surface_group):
        self._surface_group = surface_group

        # CoordinateSystemFactory requires access to SurfaceFactory attributes
        self._coordinate_factory = CoordinateSystemFactory(self)
        self._geometry_factory = GeometryFactory()
        self._material_factory = MaterialFactory()
        self._coating_factory = CoatingFactory()

    def create_surface(self, surface_type, comment, index, is_stop, material, **kwargs):
        """Creates a surface object based on the given parameters.

        Args:
            surface_type (str): The type of surface to create.
            comment (str): A comment for the surface.
            index (int): The index of the surface.
            is_stop (bool): Indicates whether the surface is a stop surface.
            material (str or tuple or BaseMaterial): The material of the surface.
            **kwargs: Additional keyword arguments for configuring the surface.

        Returns:
            Surface: The created surface object.

        Raises:
            ValueError: If the index is greater than the number of surfaces.
        """
        if index > self._surface_group.num_surfaces:
            raise ValueError("Surface index cannot be greater than number of surfaces.")

        coordinate_system = self._coordinate_factory.create(
            index, self._surface_group, **kwargs
        )

        material_pre, material_post = self._material_factory.create(
            index, material, self._surface_group
        )

        coating = self._coating_factory.create(
            kwargs.get("coating"), material_pre, material_post
        )

        is_reflective = material == "mirror"

        geometry = self._geometry_factory.create(
            surface_type, coordinate_system, **kwargs
        )

        # Special case: object surface
        if index == 0:
            if surface_type == "paraxial":
                raise ValueError("Paraxial surface cannot be the object surface.")
            return ObjectSurface(geometry, material_post, comment)

        common_params = {"aperture": kwargs.get("aperture")}

        # Create the appropriate surface type
        if surface_type == "paraxial":
            return ParaxialSurface(
                kwargs["f"],
                geometry,
                material_pre,
                material_post,
                is_stop,
                is_reflective=is_reflective,
                coating=coating,
                surface_type=surface_type,
                **common_params,
            )

        return Surface(
            geometry,
            material_pre,
            material_post,
            is_stop,
            is_reflective=is_reflective,
            coating=coating,
            surface_type=surface_type,
            comment=comment,
            **common_params,
        )
