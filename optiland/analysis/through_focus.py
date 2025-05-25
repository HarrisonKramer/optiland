"""Through-focus analysis base class.

This module provides a base class for performing through-focus analysis.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod
# import optiland.backend as be # Not used in this file


class ThroughFocusAnalysis(ABC):
    """Base class for through-focus analysis.

    This class provides the basic framework for analyses that need to be
    performed over a range of focal planes. It handles the iteration through
    different focal positions and delegates the specific analysis at each
    position to subclasses.

    Attributes:
        optic (optic.Optic): The optical system to analyze.
        delta_focus (float): The increment of focal shift in mm.
        num_steps (int): The number of focal planes to analyze before and
            after the nominal focus. For example, if `num_steps` is 5,
            analysis will be done at focal shifts from -5*`delta_focus` to
            +5*`delta_focus`, including the nominal focus (0).
        fields (list): List of resolved field coordinates (e.g., [(0,0), (0,1)])
            at which data is generated.
        wavelengths (list): List of resolved wavelength values (e.g., [0.55, 0.48])
            at which data is generated.
        results (list): A list to store the results from the analysis performed
            at each focal plane. The structure of each item in the list is
            determined by the `_perform_analysis_at_focus` method of the
            subclass.
    """

    def __init__(
        self,
        optic, # optiland.optic.Optic
        delta_focus: float = 0.1,
        num_steps: int = 5,
        fields="all",
        wavelengths="all",
    ):
        """Initializes the ThroughFocusAnalysis class.

        Args:
            optic (optiland.optic.Optic): The optical system to analyze.
            delta_focus (float, optional): The increment of focal shift in mm.
                Defaults to 0.1.
            num_steps (int, optional): The number of focal planes to analyze
                before and after the nominal focus. Defaults to 5.
            fields (list[tuple[float, float]] | str, optional): Fields for
                analysis. If "all", uses all fields from `optic.fields`.
                Otherwise, expects a list of field coordinates.
                Defaults to "all".
            wavelengths (list[float] | str, optional): Wavelengths for
                analysis. If "all", uses all wavelengths from
                `optic.wavelengths`. Otherwise, expects a list of
                wavelength values. Defaults to "all".
        """
        self.optic = optic
        self.delta_focus = delta_focus
        self.num_steps = num_steps

        if fields == "all":
            self.fields = self.optic.fields.get_field_coords()
        else:
            # Assuming 'fields' is already a list of coordinate tuples
            # if not "all". Add type checking/conversion if necessary for robustness.
            self.fields = fields

        if wavelengths == "all":
            self.wavelengths = self.optic.wavelengths.get_wavelengths()
        else:
            # Assuming 'wavelengths' is already a list of floats
            # if not "all". Add type checking/conversion if necessary for robustness.
            self.wavelengths = wavelengths

        self.results = []
        self._calculate_through_focus()

    def _calculate_through_focus(self):
        """Iterates through focal planes and performs analysis.

        This method systematically shifts the image surface position of the
        optic, calls the `_perform_analysis_at_focus` method for that focal
        position, stores the returned result, and ensures the original image
        surface position is restored after all steps.
        """
        original_focus_z = self.optic.image_surface.geometry.cs.z

        try:
            for i in range(-self.num_steps, self.num_steps + 1):
                current_delta = i * self.delta_focus
                self.optic.image_surface.geometry.cs.z = (
                    original_focus_z + current_delta
                )
                # Update the optic to ensure changes are propagated.
                self.optic.update()

                analysis_result = self._perform_analysis_at_focus(
                    current_delta
                )
                self.results.append(analysis_result)
        finally:
            # Restore the original image surface z position.
            self.optic.image_surface.geometry.cs.z = original_focus_z
            # Update the optic to ensure restoration is propagated.
            self.optic.update()

    @abstractmethod
    def _perform_analysis_at_focus(self, current_delta_focus: float):
        """Performs the specific analysis at the current focal plane.

        This abstract method must be implemented by subclasses. It defines
        the core analysis logic to be executed at each focal plane defined by
        `current_delta_focus`.

        Args:
            current_delta_focus (float): The current focal shift from the
                nominal focus, in mm.

        Returns:
            Any: The result of the analysis at the specified focal plane.
                 The type and structure of this result are determined by the
                 implementing subclass.
        """
        pass

    @abstractmethod
    def view(self):
        """Visualizes or prints the results of the through-focus analysis.

        This abstract method must be implemented by subclasses. It defines
        how the collected `self.results` (containing analysis data from all
        focal planes) should be presented to the user, for example, by
        plotting graphs or printing a formatted table.
        """
        pass
