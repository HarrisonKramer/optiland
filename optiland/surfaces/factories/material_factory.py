"""Material Factory Module

This module contains the MaterialFactory class, which is responsible for generating
material instances based on input configurations. The class abstracts material
creation, allowing seamless integration with the surface factory and other optical
system components in Optiland.

Kramer Harrison, 2025
"""

from optiland.materials import BaseMaterial, IdealMaterial, Material


class MaterialFactory:
    """Factory class for creating material instances based on input specifications.

    This class provides an interface for creating different types of materials,
    including ideal materials, predefined materials, and user-defined materials.
    """

    def __init__(self):
        self._last_material = None

    def create(self, index, material_spec, surface_group):
        """Determines the material before and after a surface based on its position.

        Args:
            index (int): The index of the surface within the surface group.
            material_spec (BaseMaterial | tuple | str): The material specification.
            surface_group (SurfaceGroup): The surface group containing the surfaces.

        Returns:
            tuple[BaseMaterial | None, BaseMaterial]:
                - The material before the surface (None for the object surface).
                - The material after the surface.
        """
        # Determine material before the surface
        if index == 0:
            material_pre = None  # Object surface has no preceding material
        elif index == surface_group.num_surfaces and self._last_material is not None:
            material_pre = self._last_material  # Image surface
        else:
            previous_surface = surface_group.surfaces[index - 1]
            material_pre = previous_surface.material_post

        # Determine material after the surface
        material_post = MaterialFactory._configure_post_material(material_spec)
        self._last_material = material_post

        # Special case for mirrors: maintain the same material before and after
        if material_spec == "mirror":
            material_post = material_pre

        return material_pre, material_post

    @staticmethod
    def _configure_post_material(material_spec):
        """Creates a post-surface material instance based on the given specification.

        Args:
            material_spec (BaseMaterial | tuple | str): The material specification.
                - If an instance of `BaseMaterial`, it is returned as-is.
                - If a tuple, it is expected to contain (name, reference) and will
                  be used to create a `Material` instance.
                - If a string, it is interpreted as a material name, with special
                  handling for 'air' and 'mirror'.

        Returns:
            BaseMaterial: The created material instance.

        Raises:
            ValueError: If an unrecognized material type is provided.
        """
        if isinstance(material_spec, BaseMaterial):
            return material_spec

        if isinstance(material_spec, tuple):
            return Material(name=material_spec[0], reference=material_spec[1])

        if isinstance(material_spec, str):
            if material_spec.lower() == "air":
                return IdealMaterial(n=1.0, k=0.0)
            if material_spec.lower() == "mirror":
                return None  # Mirror material is handled separately in surface logic.
            return Material(material_spec)

        raise ValueError(f"Unrecognized material specification: {material_spec}")
