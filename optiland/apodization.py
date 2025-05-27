"""Apodization Module.

This module defines apodization functions and classes to modify the amplitude
distribution of light across an optical system's pupil. Apodization can be
used to alter the point spread function (PSF) and other diffraction characteristics.

This module is currently a placeholder for future apodization functionality.
"""

import optiland.backend as be


class Apodization:
    """Base class for apodization functions.

    This class is intended to be subclassed to implement specific
    apodization profiles (e.g., Gaussian, Hamming, uniform).
    """

    def __init__(self):
        """Initializes the Apodization class."""
        pass

    def apply(self, pupil_coordinates):
        """Applies the apodization function to given pupil coordinates.

        This method should be overridden by subclasses to define the specific
        apodization logic.

        Args:
            pupil_coordinates (be.ndarray): A 2D array of (rho, phi) or (x, y)
                coordinates in the pupil plane.

        Returns:
            be.ndarray: An array of the same shape as pupil_coordinates,
                representing the amplitude transmission factor at each point.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the apply method.")

    def to_dict(self):
        """Serializes the apodization object to a dictionary.

        Returns:
            dict: A dictionary representation of the apodization object.
        """
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data):
        """Creates an Apodization instance from a dictionary.

        Args:
            data (dict): A dictionary containing the apodization data.

        Returns:
            Apodization: An instance of an Apodization subclass.

        Raises:
            ValueError: If the apodization type in data is unknown.
        """
        apodization_type = data.get("type")
        # This would need to be expanded as actual apodization types are added
        if apodization_type == "Apodization": # Or specific types like "GaussianApodization"
            return cls()
        # Example for a future subclass:
        # if apodization_type == "GaussianApodization":
        #     return GaussianApodization.from_dict(data)
        raise ValueError(f"Unknown apodization type: {apodization_type}")

# Example of a possible future subclass:
# class GaussianApodization(Apodization):
#     """Implements a Gaussian apodization profile."""
#     def __init__(self, sigma=1.0):
#         super().__init__()
#         self.sigma = sigma
#
#     def apply(self, pupil_coordinates):
#         """Applies Gaussian apodization.
#
#         Assumes pupil_coordinates contains radial distance rho as the first element
#         or as the x-coordinate if Cartesian.
#         """
#         if pupil_coordinates.ndim == 1: # Simple 1D radial case
#             rho_sq = pupil_coordinates**2
#         elif pupil_coordinates.shape[-1] == 2: # (x,y) or (rho, phi)
#             # Assuming first column is radial or x for simplicity here
#             rho_sq = pupil_coordinates[..., 0]**2
#         else:
#             raise ValueError("Unsupported pupil_coordinates shape for GaussianApodization.")
#         return be.exp(-rho_sq / (2 * self.sigma**2))
#
#     def to_dict(self):
#         data = super().to_dict()
#         data["sigma"] = self.sigma
#         return data
#
#     @classmethod
#     def from_dict(cls, data):
#         sigma = data.get("sigma", 1.0)
#         return cls(sigma=sigma)
