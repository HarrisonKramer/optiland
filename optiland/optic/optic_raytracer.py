# Import for type hinting
from optiland.rays import RealRays
from optiland.distribution import BaseDistribution
from typing import Union
import optiland.backend as be # For be.ndarray type hint

class OpticRayTracer:
    def __init__(self, optic):
        self._real_ray_tracer = optic.ray_tracer

    def trace(self, Hx: Union[float, be.ndarray], Hy: Union[float, be.ndarray], wavelength: float, num_rays: int = 100, distribution: Union[str, 'BaseDistribution'] = "hexapolar") -> 'RealRays':
        """Trace a distribution of rays through the optical system.

        Args:
            Hx (float or be.ndarray): The normalized x field coordinate(s).
            Hy (float or be.ndarray): The normalized y field coordinate(s).
            wavelength (float): The wavelength of the rays in microns.
            num_rays (int, optional): The number of rays to be traced.
                Defaults to 100.
            distribution (str or optiland.distribution.BaseDistribution, optional):
                The distribution of the rays. Can be a string identifier (e.g.,
                'hexapolar', 'uniform') or a Distribution object.
                Defaults to 'hexapolar'.

        Returns:
            RealRays: The RealRays object containing the traced rays.
        """
        return self._real_ray_tracer.trace(Hx, Hy, wavelength, num_rays, distribution)

    def trace_generic(self, Hx: Union[float, be.ndarray], Hy: Union[float, be.ndarray], Px: Union[float, be.ndarray], Py: Union[float, be.ndarray], wavelength: float) -> 'RealRays':
        """Trace generic rays through the optical system.

        Args:
            Hx (float or be.ndarray): The normalized x field coordinate(s).
            Hy (float or be.ndarray): The normalized y field coordinate(s).
            Px (float or be.ndarray): The normalized x pupil coordinate(s).
            Py (float or be.ndarray): The normalized y pupil coordinate(s).
            wavelength (float): The wavelength of the rays in microns.
        Returns:
            RealRays: The RealRays object containing the traced rays.
        """
        return self._real_ray_tracer.trace_generic(Hx, Hy, Px, Py, wavelength)
