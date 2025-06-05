"""Surface Factory

This module contains the SurfaceFactory class, which is used to create surface
objects based on the given parameters. The SurfaceFactory class is used by the
SurfaceGroup class to create surfaces for the optical system. The class
abstracts the creation of surface objects and allows for easy configuration of
the surface parameters.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.surfaces.factories.coating_factory import CoatingFactory
from optiland.surfaces.factories.coordinate_system_factory import (
    CoordinateSystemFactory,
)
from optiland.surfaces.factories.geometry_factory import GeometryConfig, GeometryFactory
from optiland.surfaces.factories.material_factory import MaterialFactory
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.paraxial_surface import ParaxialSurface
from optiland.surfaces.standard_surface import Surface


class SurfaceFactory:
    _surface_registry = {}

    @classmethod
    def register_surface_impl(cls, surface_type_name, handler_func):
        cls._surface_registry[surface_type_name] = handler_func

    @staticmethod
    def _create_paraxial_surface_handler(
        factory_instance,
        surface_type_str,
        comment,
        index,
        is_stop,
        material_pre,
        material_post,
        is_reflective,
        coating,
        coordinate_system,
        geometry_config,
        **kwargs,
    ):
        focal_length = kwargs["f"]
        geometry = factory_instance._geometry_factory.create(
            surface_type_str, coordinate_system, geometry_config
        )
        # ParaxialSurface expects focal_length as a positional argument
        surface_obj = ParaxialSurface(
            focal_length,
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            is_stop=is_stop,
            is_reflective=is_reflective,
            coating=coating,
            surface_type=surface_type_str,
            aperture=kwargs.get("aperture"),
        )
        return surface_obj

    @staticmethod
    def _create_standard_surface_handler(
        factory_instance,
        surface_type_str,
        comment,
        index,
        is_stop,
        material_pre,
        material_post,
        is_reflective,
        coating,
        coordinate_system,
        geometry_config,
        **kwargs,
    ):
        # Map surface_type_str from SurfaceFactory to what GeometryFactory expects
        gf_map = {
            "plane": "standard",  # GeometryFactory's _create_standard handles radius=inf as Plane
            "sphere": "standard", # GeometryFactory's _create_standard handles finite radius
            "conic": "standard",  # A conic is a StandardGeometry with a conic constant
            # "asphere" is tricky. If it means a base conic, "standard" is fine.
            # If it implies coefficients, it should be "even_asphere" or "odd_asphere".
            # Given "asphere" is registered to this standard_handler, and not a specific asphere handler,
            # assume it means a conic section that can be described by StandardGeometry.
            "asphere": "standard",
        }
        # Use mapped type if available, otherwise pass original (e.g., "even_asphere" directly)
        actual_geometry_factory_type = gf_map.get(surface_type_str, surface_type_str)

        geometry = factory_instance._geometry_factory.create(
            actual_geometry_factory_type, coordinate_system, geometry_config
        )
        surface_obj = Surface(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            is_stop=is_stop,
            is_reflective=is_reflective,
            coating=coating,
            surface_type=surface_type_str,
            comment=comment,
            aperture=kwargs.get("aperture"),
        )
        return surface_obj

    """A factory class for creating surface objects by delegating to sub-factories.
    (Docstring for __init__ and other methods remain similar)
    """
    def __init__(self, surface_group):
        self._surface_group = surface_group
        self._coordinate_factory = CoordinateSystemFactory(self)
        self._geometry_factory = GeometryFactory()
        self._material_factory = MaterialFactory()
        self._coating_factory = CoatingFactory()
        self.use_absolute_cs = False

    def create_surface(self, surface_type, comment, index, is_stop, material, **kwargs):
        if index > self._surface_group.num_surfaces:
            raise IndexError("Surface index cannot be greater than number of surfaces.")

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
            conic_x=kwargs.get("conic_x", 0.0),
            conic_y=kwargs.get("conic_y", 0.0),
            toroidal_coeffs_poly_y=kwargs.get("toroidal_coeffs_poly_y", []),
        )

        if index == 0:
            if surface_type == "paraxial": # This check needs to be here
                raise ValueError("Paraxial surface cannot be the object surface.")
            # Geometry for ObjectSurface still uses surface_type for its shape
            geometry = self._geometry_factory.create(surface_type, coordinate_system, geometry_config)
            surface_obj = ObjectSurface(geometry, material_post, comment)
            surface_obj.thickness = kwargs.get("thickness", 0.0)
            return surface_obj

        handler = self._surface_registry.get(surface_type)
        if handler:
            # Prepare kwargs for handler, avoid passing 'coating' if it was in original kwargs,
            # as it's now an explicit `coating` object.
            handler_specific_kwargs = kwargs.copy()
            if 'coating' in handler_specific_kwargs: # The original spec for coating
                del handler_specific_kwargs['coating']

            surface_obj = handler(
                factory_instance=self,
                surface_type_str=surface_type,
                comment=comment,
                index=index,
                is_stop=is_stop,
                material_pre=material_pre,
                material_post=material_post,
                is_reflective=is_reflective,
                coating=coating,
                coordinate_system=coordinate_system,
                geometry_config=geometry_config,
                **handler_specific_kwargs, # Pass modified kwargs
            )
        else:
            # Fallback to old logic OR raise error.
            # For now, let's ensure it raises error for unknown, as per previous logic.
            raise ValueError(f"Unknown or unregistered surface_type: {surface_type}")

        surface_obj.thickness = kwargs.get("thickness", 0.0)
        return surface_obj

# --- End of SurfaceFactory class definition ---

# Explicitly register ParaxialSurface handler
SurfaceFactory.register_surface_impl(
    "paraxial",
    SurfaceFactory._create_paraxial_surface_handler
)

# Explicitly register StandardSurface handler for multiple geometry types
standard_surface_handler_ref = SurfaceFactory._create_standard_surface_handler
geometry_types_for_standard_handler = [
    "standard", "plane", "sphere", "asphere", "conic", "even_asphere",
    "odd_asphere", "polynomial", "chebyshev", "biconic", "toroidal", "zernike"
]
for geo_type in geometry_types_for_standard_handler:
    SurfaceFactory.register_surface_impl(geo_type, standard_surface_handler_ref)

# Ensure other potential top-level definitions or imports are not affected if any.
# For now, assuming this is the end of the relevant part of the file.
