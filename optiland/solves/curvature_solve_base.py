import abc

import optiland.backend as be
from optiland.solves.base import BaseSolve


class CurvatureSolveBase(BaseSolve, abc.ABC):
    """Abstract base class for solves that modify surface curvature.

    This class serves as a base for solves that adjust the curvature of a
    specific optical surface to achieve a target ray angle (slope) for a
    paraxial ray (e.g., marginal or chief) emerging from that surface.

    Attributes:
        optic: The main optical system object.
        surface_idx (int): Integer index of the surface whose curvature is to
                           be modified and where the angle is targeted.
        angle (float): The target ray slope after refraction at surface_idx.
    """

    def __init__(self, optic, surface_idx: int, angle: float):
        """Initializes CurvatureSolveBase.

        Args:
            optic: The main optical system object.
            surface_idx: Integer index of the surface whose curvature is to be
                         modified and where the angle is targeted.
            angle: Float, the target ray slope after refraction at
                   surface_idx.
        """
        super().__init__()  # Call BaseSolve's __init__
        self.optic = optic
        self.surface_idx = surface_idx
        self.angle = angle

    @abc.abstractmethod
    def _get_paraxial_data_at_surface(self):
        """Fetches paraxial ray data at the specified surface.

        This method must be implemented by subclasses to return data specific
        to the ray type (marginal or chief) at self.surface_idx.

        Returns:
            tuple: A tuple containing four elements:
                - y_k (float): Ray height at self.surface_idx.
                - u_k (float): Ray slope before refraction at self.surface_idx.
                - n_k (float): Refractive index of the medium immediately
                               preceding self.surface_idx.
                - n_prime_k (float): Refractive index of the medium immediately
                                     succeeding self.surface_idx.
        """
        pass  # pragma: no cover

    def apply(self):
        """Calculates and applies the new curvature to the surface.

        This method uses the paraxial refraction formula to determine the
        necessary curvature to achieve the target ray angle. It includes
        handling for edge cases where the calculation might be invalid.
        If an invalid condition is met, the method will return without an error,
        but the curvature will not be changed. Specific error conditions should
        be checked by the caller if details are needed.
        """
        try:
            y_k, u_k, n_k, n_prime_k = self._get_paraxial_data_at_surface()
        except IndexError:
            # Silently return if surface_idx is out of bounds for ray data
            return
        except Exception:
            # Silently return for other errors during paraxial data retrieval
            return

        u_prime_k_target = self.angle

        if be.abs(y_k) < 1e-9:
            # Ray height is essentially zero.
            # Curvature change has no effect or target cannot be uniquely determined.
            # Silently return, curvature not changed.
            return

        if be.abs(n_prime_k - n_k) < 1e-9:
            # Refractive indices are the same or very close.
            # If n_k*u_k != n_prime_k*u_prime_k_target, target is impossible.
            # If n_k*u_k == n_prime_k*u_prime_k_target, no change needed.
            # Silently return, curvature not changed.
            return

        denominator = y_k * (n_prime_k - n_k)
        if be.abs(denominator) < 1e-12:
            # Denominator is too small, calculation is unstable or target unachievable.
            # Silently return, curvature not changed.
            return

        c_k = (n_k * u_k - n_prime_k * u_prime_k_target) / denominator

        if not (0 <= self.surface_idx < len(self.optic.surface_group.surfaces)):
            # surface_idx is out of bounds for the surface_group itself
            # This check is after paraxial data retrieval, which might seem late,
            # but _get_paraxial_data_at_surface might have its own checks.
            # If it passed those, this is a final safety.
            return

        self.optic.surface_group.surfaces[self.surface_idx].geometry.curvature = c_k
        # Silently update curvature

    def to_dict(self):
        """Returns a dictionary representation of the solve.

        Returns:
            dict: A dictionary representation of the solve, including its type,
                  surface_idx, and angle.
        """
        solve_dict = super().to_dict()
        solve_dict.update(
            {
                "surface_idx": self.surface_idx,
                "angle": self.angle,
            }
        )
        return solve_dict

    @classmethod
    def from_dict(cls, optic, data):
        """Reconstructs an instance of a CurvatureSolveBase subclass from data.

        Args:
            optic: The main optical system object.
            data (dict): The dictionary representation of the solve, expected
                         to contain 'surface_idx' and 'angle'.

        Returns:
            An instance of a CurvatureSolveBase subclass.

        Raises:
            TypeError: If `cls` is `CurvatureSolveBase` itself, as it's an
                       abstract class.
        """
        if cls is CurvatureSolveBase:
            raise TypeError(
                "CurvatureSolveBase is an abstract class and cannot be "
                "instantiated directly. Use a concrete subclass."
            )

        if "surface_idx" not in data or "angle" not in data:
            raise ValueError(
                "Data for CurvatureSolveBase subclass must include "
                "'surface_idx' and 'angle'."
            )

        return cls(optic, data["surface_idx"], data["angle"])
