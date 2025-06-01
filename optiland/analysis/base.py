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
                # Assuming primary_wavelength might be an object with a .value attribute
                # or just a value itself. If it's an object, .value is common.
                # If get_wavelengths() returns a list of objects, this needs adjustment too.
                # For now, let's assume primary_wavelength is directly the value or has .value
                # and get_wavelengths() returns values.
                if hasattr(self.optic, "primary_wavelength"):
                    if hasattr(self.optic.primary_wavelength, "value"):
                        self.wavelengths = [self.optic.primary_wavelength.value]
                    else:
                        self.wavelengths = [self.optic.primary_wavelength]
                else:
                    # Fallback if primary_wavelength attribute doesn't exist,
                    # though Optic class should define it.
                    # This case might indicate an issue with Optic class or assumptions.
                    raise AttributeError("Optic has no primary_wavelength defined.")
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

        # Ensure all wavelengths are floats if they are not already (e.g. Wavelength objects)
        # This step depends on what get_wavelengths() and primary_wavelength.value return.
        # Assuming they return float values directly based on typical usage.
        # If they return Wavelength objects, we would need to extract .value from each.
        # For now, this check is a placeholder for potential future refinement if types are complex.
        if self.wavelengths and not all(isinstance(w, float) for w in self.wavelengths):
            # This is a simplified check. If Wavelength objects are used,
            # they should be converted to their float .value during the
            # 'all' or 'primary' processing or when a list is passed.
            # For now, assuming float values are correctly populated.
            pass  # print("Warning: Wavelengths list might not contain all floats.")

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
