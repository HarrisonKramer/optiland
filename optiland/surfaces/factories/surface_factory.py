"""Surface Factory

This module contains the SurfaceFactory class, which is used to create surface
objects based on the given parameters. The SurfaceFactory class is used by the
SurfaceGroup class to create surfaces for the optical system. The class
abstracts the creation of surface objects and allows for easy configuration of
the surface parameters.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.surfaces.factories.coating_factory import CoatingFactory
from optiland.surfaces.factories.coordinate_system_factory import (
    CoordinateSystemFactory,
)
from optiland.surfaces.factories.geometry_factory import GeometryConfig, GeometryFactory
from optiland.surfaces.factories.material_factory import MaterialFactory
from optiland.surfaces.grating_surface import GratingSurface
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.paraxial_surface import ParaxialSurface
from optiland.surfaces.standard_surface import Surface

if TYPE_CHECKING:
    from optiland._types import SurfaceParameters, SurfaceType, Unpack
    from optiland.materials.base import BaseMaterial


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
        self.material_factory = MaterialFactory()
        self._coating_factory = CoatingFactory()

        self.use_absolute_cs = False

    def create_surface(
        self,
        surface_type: SurfaceType,
        comment: str,
        index: int | None,
        is_stop: bool,
        material: BaseMaterial | str,
        **kwargs: Unpack[SurfaceParameters],
    ):
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
            raise IndexError("Surface index cannot be greater than number of surfaces.")

        # Build coordinate system
        coordinate_system = self._coordinate_factory.create(
            index, self._surface_group, **kwargs
        )

        # Build pre and post surface materials
        material_pre, material_post = self.material_factory.create(
            index, material, self._surface_group
        )

        # Build coating
        coating = self._coating_factory.create(
            kwargs.get("coating"), material_pre, material_post
        )

        is_reflective = material == "mirror"

        # Build geometry
        geometry_config = GeometryConfig(
            radius=kwargs.get("radius", be.inf),
            conic=kwargs.get("conic", 0.0),
            coefficients=kwargs.get("coefficients", []),
            tol=kwargs.get("tol", 1e-6),
            max_iter=kwargs.get("max_iter", 100),
            norm_x=kwargs.get("norm_x", 1.0),
            norm_y=kwargs.get("norm_y", 1.0),
            norm_radius=kwargs.get("norm_radius", 1.0),
            radius_x=kwargs.get("radius_x", be.inf),
            radius_y=kwargs.get("radius_y", be.inf),
            grating_order=kwargs.get("grating_order", 0),
            grating_period=kwargs.get("grating_period", be.inf),
            groove_orientation_angle=kwargs.get("groove_orientation_angle", 0.0),
            conic_x=kwargs.get("conic_x", 0.0),
            conic_y=kwargs.get("conic_y", 0.0),
            toroidal_coeffs_poly_y=kwargs.get("toroidal_coeffs_poly_y", []),
            zernike_type=kwargs.get("zernike_type", "fringe"),
            radial_terms=kwargs.get("radial_terms"),
            freeform_coeffs=kwargs.get("freeform_coeffs"),
            forbes_norm_radius=kwargs.get("forbes_norm_radius", 1.0),
        )

        geometry = self._geometry_factory.create(
            surface_type, coordinate_system, geometry_config
        )

        # Special case: object surface
        if index == 0:
            if surface_type == "paraxial":
                raise ValueError("Paraxial surface cannot be the object surface.")
            surface_obj = ObjectSurface(geometry, material_post, comment)
            surface_obj.thickness = kwargs.get("thickness", 0.0)
            return surface_obj

        # Create the appropriate surface type
        if surface_type == "paraxial":
            surface_obj = ParaxialSurface(
                kwargs["f"],
                geometry,
                material_pre,
                material_post,
                is_stop,
                is_reflective=is_reflective,
                coating=coating,
                surface_type=surface_type,
                aperture=kwargs.get("aperture"),
            )
            surface_obj.thickness = kwargs.get("thickness", 0.0)
            return surface_obj

        # Create the appropriate surface type
        if surface_type == "grating":
            surface_obj = GratingSurface(
                geometry,
                material_pre,
                material_post,
                is_stop,
                is_reflective=is_reflective,
                coating=coating,
                surface_type=surface_type,
                aperture=kwargs.get("aperture"),
            )
            surface_obj.thickness = kwargs.get("thickness", 0.0)
            return surface_obj

        # Standard surface - `surface_type` indicates geometrical shape of surface
        surface_obj = Surface(
            geometry,
            material_pre,
            material_post,
            is_stop,
            is_reflective=is_reflective,
            coating=coating,
            surface_type=surface_type,
            comment=comment,
            aperture=kwargs.get("aperture"),
        )

        # Add the thickness as an attribute to the surface
        surface_obj.thickness = kwargs.get("thickness", 0.0)
        return surface_obj
