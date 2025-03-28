"""Polarized Rays

This module contains the `PolarizedRays` class, which represents a class for
polarized rays in three-dimensional space. The class inherits from the
`RealRays` class.

Kramer Harrison, 2024
"""

import numpy as np

from optiland.rays.polarization_state import PolarizationState
from optiland.rays.real_rays import RealRays


class PolarizedRays(RealRays):
    """Represents a class for polarized rays in three-dimensional space.

    Inherits from the `RealRays` class.

    Attributes:
        x (ndarray): The x-coordinates of the rays.
        y (ndarray): The y-coordinates of the rays.
        z (ndarray): The z-coordinates of the rays.
        L (ndarray): The x-components of the direction vectors of the rays.
        M (ndarray): The y-components of the direction vectors of the rays.
        N (ndarray): The z-components of the direction vectors of the rays.
        i (ndarray): The intensity of the rays.
        w (ndarray): The wavelength of the rays.
        opd (ndarray): The optical path length of the rays.
        p (np.ndarray): Array of polarization matrices of the rays.

    Methods:
        get_output_field(E: np.ndarray) -> np.ndarray:
            Compute the output electric field given the input electric field.
        update_intensity(state: PolarizationState):
            Update the ray intensity based on the polarization state.
        update(jones_matrix: np.ndarray = None):
            Update the polarization matrices after interaction with a surface.
        _get_3d_electric_field(state: PolarizationState) -> np.ndarray:
            Get the 3D electric fields given the polarization state and
            initial rays.

    """

    def __init__(self, x, y, z, L, M, N, intensity, wavelength):
        super().__init__(x, y, z, L, M, N, intensity, wavelength)

        self.p = np.tile(np.eye(3), (self.x.size, 1, 1))
        self._i0 = intensity.copy()
        self._L0 = L.copy()
        self._M0 = M.copy()
        self._N0 = N.copy()

    def get_output_field(self, E: np.ndarray) -> np.ndarray:
        """Compute the output electric field given the input electric field.

        Args:
            E (np.ndarray): The input electric field as a numpy array.

        Returns:
            np.ndarray: The computed output electric field as a numpy array.

        """
        return np.squeeze(np.matmul(self.p, E[:, :, np.newaxis]), axis=2)

    def update_intensity(self, state: PolarizationState):
        """Update ray intensity based on polarization state.

        Args:
            state (PolarizationState): The polarization state of the ray.

        """
        if state.is_polarized:
            E0 = self._get_3d_electric_field(state)
            E1 = self.get_output_field(E0)
            self.i = np.sum(np.abs(E1) ** 2, axis=1)
        else:
            # Local x-axis field
            state_x = PolarizationState(
                is_polarized=True,
                Ex=1.0,
                Ey=0.0,
                phase_x=0.0,
                phase_y=0.0,
            )
            E0_x = self._get_3d_electric_field(state_x)
            E1_x = self.get_output_field(E0_x)

            # Local y-axis field
            state_y = PolarizationState(
                is_polarized=True,
                Ex=0.0,
                Ey=1.0,
                phase_x=0.0,
                phase_y=0.0,
            )
            E0_y = self._get_3d_electric_field(state_y)
            E1_y = self.get_output_field(E0_y)

            # average two orthogonal polarizations to get mean intensity,
            # scale by initial ray intensity
            self.i = (
                (np.sum(np.abs(E1_x) ** 2, axis=1) + np.sum(np.abs(E1_y) ** 2, axis=1))
                * self._i0
                / 2
            )

    def update(self, jones_matrix: np.ndarray = None):
        """Update polarization matrices after interaction with surface.

        Args:
            jones_matrix (np.ndarray, optional): Jones matrix representing the
                interaction with the surface. If not provided, the
                polarization matrix is computed assuming an identity matrix.

        """
        # merge k-vector components into matrix for speed
        k0 = np.array([self.L0, self.M0, self.N0]).T
        k1 = np.array([self.L, self.M, self.N]).T

        # find s-component
        s = np.cross(k0, k1)
        mag = np.linalg.norm(s, axis=1)

        # handle case when mag = 0 (i.e., k0 parallel to k1)
        if np.any(mag == 0):
            s[mag == 0] = np.cross(k0[mag == 0], np.array([1.0, 0.0, 0.0]))
            mag = np.linalg.norm(s, axis=1)

        s /= mag[:, np.newaxis]

        # find p-component pre and post surface
        p0 = np.cross(k0, s)
        p1 = np.cross(k1, s)

        # othogonal transformation matrices
        o_in = np.stack((s, p0, k0), axis=1)
        o_out = np.stack((s, p1, k1), axis=2)

        # compute polarization matrix for surface
        if jones_matrix is None:
            p = np.matmul(o_out, o_in)
        else:
            p = np.einsum("nij,njk,nkl->nil", o_out, jones_matrix, o_in)

        # update polarization matrices of rays
        self.p = np.matmul(p, self.p)

    def _get_3d_electric_field(self, state: PolarizationState) -> np.ndarray:
        """Get 3D electric fields given polarization state and initial rays.

        Args:
            state (PolarizationState): The polarization state of the rays.

        Returns:
            np.ndarray: The 3D electric fields.

        """
        k = np.array([self._L0, self._M0, self._N0]).T

        # TODO - efficiently handle case when k parallel to x-axis
        x = np.array([1.0, 0.0, 0.0])
        p = np.cross(k, x)

        norms = np.linalg.norm(p, axis=1)
        if np.any(norms == 0):
            raise ValueError("k-vector parallel to x-axis is not currently supported.")

        p /= norms[:, np.newaxis]

        s = np.cross(p, k)

        E = (
            state.Ex * np.exp(1j * state.phase_x) * s
            + state.Ey * np.exp(1j * state.phase_y) * p
        )

        return E
