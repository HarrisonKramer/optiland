"""Analysis Base Module

This module contains the abstract base class for all analysis
classes in the Optiland package.

Kramer Harrison, 2025
"""

import abc


class BaseAnalysis(abc.ABC):
    """Base class for all analysis routines.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Can be 'all', 'primary', or a list of wavelength values.
            Defaults to 'all'.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The list of wavelengths (in µm) being analyzed.
        data: The generated analysis data. This is populated by the
              `_generate_data` method implemented by subclasses.
    """

    def __init__(self, optic, wavelengths="all"):
        self.optic = optic

        if isinstance(wavelengths, str):
            if wavelengths == "all":
                self.wavelengths = self.optic.wavelengths.get_wavelengths()
            elif wavelengths == "primary":
                self.wavelengths = [self.optic.primary_wavelength]
            else:
                raise ValueError(
                    "Invalid wavelength string. Must be 'all' or 'primary'."
                )
        elif isinstance(wavelengths, list):
            self.wavelengths = wavelengths
        else:
            raise TypeError(
                "Wavelengths must be a string ('all', 'primary') or a list."
            )

        self.data = self._generate_data()

    @abc.abstractmethod
    def _generate_data(self):
        """Abstract method to generate analysis-specific data.

        This method must be implemented by subclasses. It should perform
        the necessary calculations and return the data to be stored in
        `self.data`.
        """
        pass

    @abc.abstractmethod
    def view(self, figsize=None, **kwargs):
        """Abstract method to visualize the analysis data.

        This method must be implemented by subclasses.

        Args:
            figsize (tuple, optional): The figure size for the plot.
            **kwargs: Additional keyword arguments for customization.
        """
        pass
