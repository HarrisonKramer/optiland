from abc import ABC, abstractmethod


class BaseMaterial(ABC):
    """
    Base class for materials.

    This class defines the interface for material properties such as
    refractive index (n) and extinction coefficient (k). It also provides a
    method to calculate the Abbe number.

    Subclasses of BaseMaterial should implement the abstract methods `n` and
    `k` to provide specific material properties.

    Attributes:
        None

    Methods:
        n(wavelength): Abstract method to calculate the refractive index at a
            given wavelength in microns.
        k(wavelength): Abstract method to calculate the extinction coefficient
            at a given wavelength in microns.
        abbe(): Method to calculate the Abbe number of the material.
    """
    @abstractmethod
    def n(self, wavelength):
        pass  # pragma: no cover

    @abstractmethod
    def k(self, wavelength):
        pass  # pragma: no cover

    def abbe(self):
        """
        Calculate the Abbe number of the material.

        The Abbe number is a measure of the dispersion of a material, defined
        as the difference in refractive index between the D-line
        (wavelength = 0.5893 μm) and the F-line (wavelength = 0.4861 μm)
        divided by the difference between the F-line and the C-line
        (wavelength = 0.6563 μm).

        Returns:
            The Abbe number of the material.
        """
        nD = self.n(0.5875618)
        nF = self.n(0.4861327)
        nC = self.n(0.6562725)
        return (nD - 1) / (nF - nC)

    def PgF(self):
        """
        Calculate the relative partial dispersion of the material at g and F
        lines.

        Returns:
            The relative partial dispersion of the material for the g and F
                lines.
        """
        nF = self.n(0.4861327)
        ng = self.n(0.4358343)
        nC = self.n(0.6562725)
        if nF == nC:
            return 0
        return (ng - nF) / (nF - nC)
