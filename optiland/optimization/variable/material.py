"""Material Variable Module

This module contains the MaterialVariable class, which represents a variable for
the material at a specific surface such as a glass.
The variable can be used in optimization problems to optimize the material
at a specific surface.
"""

from optiland.materials.base import BaseMaterial
from optiland.optimization.variable.base import VariableBehavior
from optiland.surfaces.factories.material_factory import MaterialFactory


class MaterialVariable(VariableBehavior):
    """Represents a variable for the material at a specific surface.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The surface number where the variable is applied.
            calculated.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        optic,
        surface_number: int,
        glass_selection: list[str],
        apply_scaling: bool = False,
        **kwargs,
    ):
        super().__init__(optic, surface_number, apply_scaling=apply_scaling, **kwargs)
        self.glass_selection = glass_selection

    def get_value(self) -> str:
        """Returns the name of the material at the specified surface.

        Returns:
            str: The material name.

        """
        surface = self.optic.surface_group.surfaces[self.surface_number]
        return surface.material_post.name

    def update_value(self, new_value: str) -> None:
        """Updates the material at the specified surface.

        Args:
            new_value (float): The new material name.

        """
        material_post: BaseMaterial = MaterialFactory._configure_post_material(
            new_value
        )
        self.optic.set_material(material_post, self.surface_number)

    def scale(self, value):
        return value

    def __str__(self) -> str:
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Material, Surface {self.surface_number}"
