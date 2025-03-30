import numpy as np

from optiland.distribution import create_distribution
from optiland.rays import PolarizedRays, RayGenerator


class RealRayTracer:
    def __init__(self, optic):
        self.optic = optic
        self.ray_generator = RayGenerator(optic)

    def trace(self, Hx, Hy, wavelength, num_rays=100, distribution="hexapolar"):
        """Trace a distribution of rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            wavelength (float): The wavelength of the rays.
            num_rays (int, optional): The number of rays to be traced. Defaults
                to 100.
            distribution (str or Distribution, optional): The distribution of
                the rays. Defaults to 'hexapolar'.

        Returns:
            RealRays: The RealRays object containing the traced rays."
        """
        self._validate_normalized_coordinates(Hx, Hy, "field")

        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        Px = distribution.x
        Py = distribution.y

        rays = self.ray_generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        self.surface_group.trace(rays)

        if isinstance(rays, PolarizedRays):
            rays.update_intensity(self.polarization_state)

        # update ray intensity
        self.surface_group.intensity[-1, :] = rays.i

        return rays

    def trace_generic(self, Hx, Hy, Px, Py, wavelength):
        """Trace generic rays through the optical system.

        Args:
            Hx (float or numpy.ndarray): The normalized x field coordinate.
            Hy (float or numpy.ndarray): The normalized y field coordinate.
            Px (float or numpy.ndarray): The normalized x pupil coordinate.
            Py (float or numpy.ndarray): The normalized y pupil coordinate
            wavelength (float): The wavelength of the rays.

        """
        self._validate_normalized_coordinates(Hx, Hy, "field")
        self._validate_normalized_coordinates(Px, Py, "pupil")

        vx, vy = self.fields.get_vig_factor(Hx, Hy)

        Px *= 1 - vx
        Py *= 1 - vy

        # assure all variables are arrays of the same size
        max_size = max([np.size(arr) for arr in [Hx, Hy, Px, Py]])
        Hx, Hy, Px, Py = [
            (
                np.full(max_size, value)
                if isinstance(value, (float, int))
                else value
                if isinstance(value, np.ndarray)
                else None
            )
            for value in [Hx, Hy, Px, Py]
        ]

        rays = self.ray_generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        rays = self.surface_group.trace(rays)

        # update intensity
        self.surface_group.intensity[-1, :] = rays.i

        return rays

    def _validate_normalized_coordinates(self, x, y, coord_type="field"):
        """Validate that normalized coordinates are within the range (-1, 1).

        Args:
            x (float or numpy.ndarray): The normalized x coordinate.
            y (float or numpy.ndarray): The normalized y coordinate.
            coord_type (str): The type of coordinates being validated
                              ('field' or 'pupil').

        Raises:
            ValueError: If the coordinates are not within the range (-1, 1).
        """
        if not np.all((x >= -1) & (x <= 1)) or not np.all((y >= -1) & (y <= 1)):
            raise ValueError(
                f"Normalized {coord_type} coordinates must be within (-1, 1)"
            )
