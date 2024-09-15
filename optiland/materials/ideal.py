from optiland.materials.base import BaseMaterial


class IdealMaterial(BaseMaterial):
    """
    Represents an ideal material with a fixed refractive index and absorption
    coefficient for all wavelengths.

    Attributes:
        index (float): The refractive index of the material.
        absorp (float): The absorption coefficient of the material.
    """

    def __init__(self, n, k=0):
        self.index = n
        self.absorp = k

    def n(self, wavelength):
        """
        Returns the refractive index of the material.

        Args:
            wavelength (float): The wavelength of light in microns.

        Returns:
            float: The refractive index of the material.
        """
        return self.index

    def k(self, wavelength):
        """
        Returns the absorption coefficient of the material.

        Args:
            wavelength (float): The wavelength of light in microns.

        Returns:
            float: The absorption coefficient of the material.
        """
        return self.absorp
