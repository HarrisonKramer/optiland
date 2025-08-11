"""
This module defines the `Wavefront` class, which is designed to analyze the
wavefront of an optical system.

Kramer Harrison, 2024
"""

from optiland.distribution import create_distribution

from .strategy import create_strategy


class Wavefront:
    """Performs wavefront analysis on an optical system.

    This class acts as a high-level controller that delegates the complex
    work of wavefront analysis to a specified strategy (e.g., 'chief_ray' or
    'best_fit'). It computes ray intersection points with the exit pupil,
    the optical path difference (OPD), ray intensities, and the radius of
    curvature of the reference sphere.

    Args:
        optic (Optic): The optical system to analyze.
        fields (str or list[tuple[float, float]]): The fields to analyze.
            Can be "all" to use all fields defined in the optic.
        wavelengths (str or list[float]): The wavelengths to analyze. Can be
            "all" for all wavelengths or "primary" for the primary wavelength.
        num_rays (int): The number of rays to use for pupil sampling.
        distribution (str or Distribution): The ray distribution pattern. Can
            be a name (e.g., "hexapolar") or a Distribution object.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray" and "best_fit". Defaults to "chief_ray".

    Attributes:
        data (dict): A dictionary containing the computed `WavefrontData` for
            each (field, wavelength) pair.
    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelengths="all",
        num_rays=12,
        distribution="hexapolar",
        strategy="chief_ray",
        **kwargs,
    ):
        self.optic = optic
        self.fields = self._resolve_fields(fields)
        self.wavelengths = self._resolve_wavelengths(wavelengths)
        self.num_rays = num_rays
        self.distribution = self._resolve_distribution(distribution, self.num_rays)

        self.strategy = create_strategy(
            strategy_name=strategy,
            optic=self.optic,
            distribution=self.distribution,
            **kwargs,
        )

        self.data = {}
        self._generate_data()

    def get_data(self, field, wl):
        """Retrieves precomputed wavefront data for a field and wavelength.

        Args:
            field (tuple[float, float]): The field coordinates.
            wl (float): The wavelength.

        Returns:
            WavefrontData: A data container with the computed wavefront results.
        """
        return self.data[(field, wl)]

    def _resolve_fields(self, fields):
        """Resolves field coordinates from the input specification."""
        if fields == "all":
            return self.optic.fields.get_field_coords()
        return fields

    def _resolve_wavelengths(self, wavelengths):
        """Resolves wavelengths from the input specification."""
        if wavelengths == "all":
            return self.optic.wavelengths.get_wavelengths()
        if wavelengths == "primary":
            return [self.optic.primary_wavelength]
        return wavelengths

    def _resolve_distribution(self, dist, num_rays):
        """Resolves the pupil distribution from the input specification."""
        if isinstance(dist, str):
            dist_obj = create_distribution(dist)
            dist_obj.generate_points(num_rays)
            return dist_obj
        return dist

    def _generate_data(self):
        """Generates wavefront data for all specified fields and wavelengths.

        This method iterates through each field and wavelength pair and
        delegates the computation to the selected strategy object.
        """
        for field in self.fields:
            for wl in self.wavelengths:
                self.data[(field, wl)] = self.strategy.compute_wavefront_data(field, wl)
