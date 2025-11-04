"""Material Factory Module

This module contains the MaterialFactory class, which is responsible for generating
material instances based on input configurations. The class abstracts material
creation, allowing seamless integration with the surface factory and other optical
system components in Optiland.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import Union

from optiland.materials import BaseMaterial, IdealMaterial, Material


class MaterialFactory:
    """A stateless factory for creating material instances from specifications."""

    @staticmethod
    def create(
        material_spec: Union[BaseMaterial, tuple, str, None]
    ) -> Union[BaseMaterial, None]:
        """Creates a material instance based on the given specification.

        This is a pure function that converts a material specification into a
        material object without relying on any external state.

        Args:
            material_spec: The material specification.
                - If an instance of `BaseMaterial`, it is returned as-is.
                - If a tuple, it is expected to contain (name, reference) and will
                  be used to create a `Material` instance.
                - If a string, it is interpreted as a material name, with special
                  handling for 'air' and 'mirror'.
                - If None, returns None.

        Returns:
            The created material instance, or None.

        Raises:
            ValueError: If an unrecognized material type is provided.
        """
        if material_spec is None:
            return None

        if isinstance(material_spec, BaseMaterial):
            return material_spec

        if isinstance(material_spec, tuple):
            return Material(name=material_spec[0], reference=material_spec[1])

        if isinstance(material_spec, str):
            if material_spec.lower() == "air":
                return IdealMaterial(n=1.0, k=0.0)
            if material_spec.lower() == "mirror":
                # The "mirror" case implies the material_post is the same as
                # material_pre. The state manager (SurfaceGroup) is responsible
                # for this logic. The factory simply returns None.
                return None
            return Material(material_spec)

        raise ValueError(f"Unrecognized material specification: {material_spec}")
