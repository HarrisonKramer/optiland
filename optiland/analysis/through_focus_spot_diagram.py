"""Through Focus Spot Diagram Analysis Module.

This module provides a class for performing through-focus spot diagram
analysis, calculating the spot diagram at various focal planes.

"""

from typing import Literal

from optiland.analysis.spot_diagram import SpotDiagram
from optiland.analysis.through_focus import ThroughFocusAnalysis


class ThroughFocusSpotDiagram(ThroughFocusAnalysis):
    """Performs spot diagram analysis over a range of focal planes.

    This class extends `ThroughFocusAnalysis` to specifically calculate and
    report RMS spot radii from spot diagrams at various focal positions.
    It utilizes the `SpotDiagram` class for the core calculations at each
    focal plane.

    Attributes:
        optic (optiland.optic.Optic): The optical system being analyzed.
        delta_focus (float): The focal shift increment in mm.
        num_steps (int): Number of focal planes analyzed before and after
            the nominal focus.
        fields (list): Resolved list of field coordinates for analysis.
        wavelengths (list): Resolved list of wavelengths for analysis.
        num_rings (int): Number of rings for pupil sampling in the
            `SpotDiagram` calculation.
        distribution (str): Pupil sampling distribution type (e.g.,
            'hexapolar', 'random') for `SpotDiagram`.
        coordinates (Literal["global", "local"]): Coordinate system used for
            spot data generation within `SpotDiagram`.
        results (list[dict[float, list[float]]]): A list where each item is a
            dictionary. Each dictionary corresponds to a single focal plane
            and maps the delta focus (float, in mm) to a list of RMS spot
            radii (list of floats, in mm). Each RMS spot radius in the list
            corresponds to a field defined in `self.fields`, calculated at the
            primary wavelength.
    """

    def __init__(
        self,
        optic,
        delta_focus: float = 0.1,
        num_steps: int = 5,
        fields="all",
        wavelengths="all",
        num_rings: int = 6,
        distribution: str = "hexapolar",
        coordinates: Literal["global", "local"] = "local",
    ):
        """Initializes the ThroughFocusSpotDiagram analysis.

        Args:
            optic (optiland.optic.Optic): The optical system to analyze.
            delta_focus (float, optional): The increment of focal shift in mm.
                Defaults to 0.1.
            num_steps (int, optional): The number of focal planes to analyze
                on either side of the nominal focus. Defaults to 5. Must be in
                range [1, 7].
            fields (list[tuple[float,float]] | str, optional): Fields for
                analysis. If "all", uses all fields from `optic.fields`.
                Otherwise, expects a list of field coordinates.
                Defaults to "all".
            wavelengths (list[float] | str, optional): Wavelengths for
                analysis. If "all", uses all wavelengths from
                `optic.wavelengths`. Otherwise, expects a list of
                wavelength values. Defaults to "all".
            num_rings (int, optional): Number of rings for pupil sampling in
                the `SpotDiagram` calculation. Defaults to 6.
            distribution (str, optional): Pupil sampling distribution type for
                `SpotDiagram` (e.g., 'hexapolar', 'random').
                Defaults to "hexapolar".
            coordinates (Literal["global", "local"], optional): Coordinate
                system for spot data generation in `SpotDiagram`.
                Defaults to "local".
        """
        super().__init__(
            optic,
            delta_focus=delta_focus,
            num_steps=num_steps,
            fields=fields,
            wavelengths=wavelengths,
        )
        self.num_rings = num_rings
        self.distribution = distribution
        if coordinates not in ["global", "local"]:
            raise ValueError("Coordinates must be 'global' or 'local'.")
        self.coordinates = coordinates

    def _perform_analysis_at_focus(self):
        """Calculates RMS spot radii at the current focal plane.

        This method is called by the base class for each focal step. It
        instantiates a `SpotDiagram` object for the optic's current focal
        state, calculates the RMS spot radius for each specified field at the
        primary wavelength, and returns this data.

        Note:
            This implementation re-instantiates `SpotDiagram` for each focal
            step, which involves recalculating ray data. For high-performance
            needs, optimizing this by directly accessing or reusing ray tracing
            functionality might be considered.

        Returns:
            list: a list of spot diagram data, including intersection points and
                intensity
        """
        spot_diagram_at_focus = SpotDiagram(
            self.optic,
            fields=self.fields,
            wavelengths=self.wavelengths,
            num_rings=self.num_rings,
            distribution=self.distribution,
            coordinates=self.coordinates,
        )
        return spot_diagram_at_focus.data

    def view(self):
        """Prints the through-focus RMS spot radius results to the console.

        Outputs the calculated RMS spot radius for each field (at the primary
        wavelength) at each evaluated focal plane.
        """
        pass
