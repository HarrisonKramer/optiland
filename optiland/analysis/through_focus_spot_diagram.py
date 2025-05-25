from typing import Literal

import optiland.backend as be
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
        optic,  # optiland.optic.Optic
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
                on either side of the nominal focus. Defaults to 5.
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

    def _perform_analysis_at_focus(self, current_delta_focus: float):
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

        Args:
            current_delta_focus (float): The current focal shift from nominal,
                in mm.

        Returns:
            dict[float, list[float]]: A dictionary mapping the
                `current_delta_focus` to a list of RMS spot radii (float,
                in mm). Each radius corresponds to a field in `self.fields`,
                calculated at the primary wavelength.
        """
        spot_diagram_at_focus = SpotDiagram(
            self.optic,
            fields=self.fields,  # Use fields resolved by the base class
            wavelengths=self.wavelengths,  # Use wavelengths resolved by base
            num_rings=self.num_rings,
            distribution=self.distribution,
            coordinates=self.coordinates,
        )

        # SpotDiagram.rms_spot_radius() returns: list[list[float]]
        # Outer list: per field; Inner list: per wavelength for that field.
        all_rms_radii = spot_diagram_at_focus.rms_spot_radius()

        primary_wavelength_idx = self.optic.wavelengths.primary_index

        rms_radii_primary_wavelength = []
        for field_idx in range(len(self.fields)):
            rms_radii_primary_wavelength.append(
                all_rms_radii[field_idx][primary_wavelength_idx]
            )

        return {current_delta_focus: rms_radii_primary_wavelength}

    def view(self):
        """Prints the through-focus RMS spot radius results to the console.

        Outputs the calculated RMS spot radius for each field (at the primary
        wavelength) at each evaluated focal plane.
        """
        for result_at_focus in self.results:
            for delta_f, rms_values in result_at_focus.items():
                print(f"  Delta Focus: {delta_f:.3f} mm")
                for i, rms_val in enumerate(rms_values):
                    field_coord = self.fields[i]
                    # Ensure rms_val is a simple float for printing,
                    # converting from backend tensor if necessary.
                    rms_val_float = (
                        be.to_numpy(rms_val).item()
                        if hasattr(rms_val, "item")
                        else float(rms_val)
                    )
                    print(
                        f"    Field ({field_coord[0]:.2f}, "
                        f"{field_coord[1]:.2f}): {rms_val_float:.6f} mm"
                    )
