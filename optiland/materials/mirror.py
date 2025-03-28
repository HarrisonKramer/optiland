"""Mirror Material

This module contains the Mirror class, which represents a mirror material.
This class inherits from the IdealMaterial class and is used to define the
optical properties of a mirror in the Optiland system.

Kramer Harrison, 2024
"""

from optiland.materials.ideal import IdealMaterial


class Mirror(IdealMaterial):
    """Represents a mirror material.

    Inherits from the IdealMaterial class.

    Attributes:
        n (float): The refractive index of the material.
        k (float): The extinction coefficient of the material.

    """

    def __init__(self):
        super().__init__(n=-1.0, k=0.0)

    @classmethod
    def from_dict(cls, data):
        """Create a mirror material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the mirror material.

        Returns:
            Mirror: The mirror material.

        """
        return Mirror()
