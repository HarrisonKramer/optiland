"""Optiland Rays Module

This module defines classes for representing and manipulating rays in optical
simulations. It includes functionality for translating and rotating rays in
three-dimensional space, as well as initializing rays with specific properties
such as position, direction, energy, and wavelength.

Kramer Harrison, 2024
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


class BaseRays:
    """
    Base class for rays in a 3D space.

    Attributes:
        x (float): x-coordinate of the ray.
        y (float): y-coordinate of the ray.
        z (float): z-coordinate of the ray.
    """

    def translate(self, dx: float, dy: float, dz: float):
        """
        Shifts the rays in the x, y, and z directions.

        Args:
            dx (float): The amount to shift the rays in the x direction.
            dy (float): The amount to shift the rays in the y direction.
            dz (float): The amount to shift the rays in the z direction.
        """
        self.x += dx
        self.y += dy
        self.z += dz

    def _process_input(self, data):
        """
        Process the input data and convert it into a 1-dimensional NumPy array
        of floats.

        Parameters:
            data (int, float, or np.ndarray): The input data to be processed.

        Returns:
            np.ndarray: The processed data as a 1-dimensional NumPy array of
                floats.

        Raises:
            ValueError: If the input data type is not supported (must be a
                scalar or a NumPy array).
        """
        if isinstance(data, (int, float)):
            return np.array([data], dtype=float)
        elif isinstance(data, np.ndarray):
            return np.ravel(data).astype(float)
        else:
            raise ValueError('Unsupported input type. Must be a scalar or a '
                             'NumPy array.')


class RealRays(BaseRays):
    """
    Represents a collection of real rays in 3D space.

    Attributes:
        x (ndarray): The x-coordinates of the rays.
        y (ndarray): The y-coordinates of the rays.
        z (ndarray): The z-coordinates of the rays.
        L (ndarray): The x-components of the direction vectors of the rays.
        M (ndarray): The y-components of the direction vectors of the rays.
        N (ndarray): The z-components of the direction vectors of the rays.
        e (ndarray): The energy of the rays.
        w (ndarray): The wavelength of the rays.
        opd (ndarray): The optical path length of the rays.

    Methods:
        rotate_x(rx: float): Rotate the rays about the x-axis.
        rotate_y(ry: float): Rotate the rays about the y-axis.
        rotate_z(rz: float): Rotate the rays about the z-axis.
        propagate(t: float): Propagate the rays a distance t.
        clip(condition): Clip the rays based on a condition.
    """

    def __init__(self, x, y, z, L, M, N, energy, wavelength):
        self.x = self._process_input(x)
        self.y = self._process_input(y)
        self.z = self._process_input(z)
        self.L = self._process_input(L)
        self.M = self._process_input(M)
        self.N = self._process_input(N)
        self.e = self._process_input(energy)
        self.w = self._process_input(wavelength)
        self.opd = np.zeros_like(self.x)

    def rotate_x(self, rx: float):
        """Rotate the rays about the x-axis."""
        y = self.y * np.cos(rx) - self.z * np.sin(rx)
        z = self.y * np.sin(rx) + self.z * np.cos(rx)
        m = self.M * np.cos(rx) - self.N * np.sin(rx)
        n = self.M * np.sin(rx) + self.N * np.cos(rx)
        self.y = y
        self.z = z
        self.M = m
        self.N = n

    def rotate_y(self, ry: float):
        """Rotate the rays about the y-axis."""
        x = self.x * np.cos(ry) + self.z * np.sin(ry)
        z = -self.x * np.sin(ry) + self.z * np.cos(ry)
        L = self.L * np.cos(ry) + self.N * np.sin(ry)
        n = -self.L * np.sin(ry) + self.N * np.cos(ry)
        self.x = x
        self.z = z
        self.L = L
        self.N = n

    def rotate_z(self, rz: float):
        """Rotate the rays about the z-axis."""
        x = self.x * np.cos(rz) - self.y * np.sin(rz)
        y = self.x * np.sin(rz) + self.y * np.cos(rz)
        L = self.L * np.cos(rz) - self.M * np.sin(rz)
        m = self.L * np.sin(rz) + self.M * np.cos(rz)
        self.x = x
        self.y = y
        self.L = L
        self.M = m

    def propagate(self, t: float):
        """Propagate the rays a distance t."""
        self.x += t * self.L
        self.y += t * self.M
        self.z += t * self.N

    def clip(self, condition):
        """Clip the rays based on a condition."""
        self.e[condition] = 0.0


class ParaxialRays(BaseRays):
    """
    Class representing paraxial rays in an optical system.

    Attributes:
        y (array-like): The y-coordinate of the rays.
        u (array-like): The slope of the rays.
        z (array-like): The z-coordinate of the rays.
        wavelength (array-like): The wavelength of the rays.

    Methods:
        propagate(t): Propagates the rays by a given distance.
    """

    def __init__(self, y, u, z, wavelength):
        self.y = self._process_input(y)
        self.z = self._process_input(z)
        self.u = self._process_input(u)
        self.x = np.zeros_like(self.y)
        self.e = np.ones_like(self.y)
        self.w = self._process_input(wavelength)

    def propagate(self, t: float):
        """
        Propagates the rays by a given distance.

        Args:
            t (float): The distance to propagate the rays.
        """
        self.z += t
        self.y += t * self.u


@dataclass
class PolarizationState:
    is_polarized: bool = False
    Ex: Optional[float] = None
    Ey: Optional[float] = None
    phase_x: Optional[float] = None
    phase_y: Optional[float] = None

    def __post_init__(self):
        if not self.is_polarized:
            if (self.Ex is not None
                    or self.Ey is not None
                    or self.phase_x is not None
                    or self.phase_y is not None):
                raise ValueError('Invalid polarization state. When state is '
                                 'not polarized, Ex, Ey, phase_x, and phase_y '
                                 'must be None.')


class PolarizedRays(RealRays):

    def __init__(self, x, y, z, L, M, N, energy, wavelength):
        super().__init__(x, y, z, L, M, N, energy, wavelength)

        # compute nominal polarization matrix, scaled to match intial energy
        # TODO - update scaling method for intiial energy, incl. apodization
        self.p = np.tile(np.eye(3), (self.x.size, 1, 1))
        p_init = np.sqrt(self.e / 2)
        self.p[:, 0, 0] = p_init
        self.p[:, 1, 1] = p_init

    def get_output_field(self, E: np.ndarray):
        """Compute output electric field given input electric field."""
        return self.p @ E

    def update_energy(self, state: PolarizationState):
        """Update ray energy based on polarization state."""
        if state.is_polarized:
            E0 = self._get_3d_electric_field(state)
            E1 = self.get_output_field(E0)
            self.e = np.abs(E1)**2
        else:
            state_x = PolarizationState(is_polarized=True, Ex=1.0, Ey=0.0,
                                        phase_x=0.0, phase_y=0.0)
            state_y = PolarizationState(is_polarized=True, Ex=0.0, Ey=1.0,
                                        phase_x=0.0, phase_y=0.0)
            E0_x = self._get_3d_electric_field(state_x)
            E0_y = self._get_3d_electric_field(state_y)
            E1_x = self.get_output_field(E0_x)
            E1_y = self.get_output_field(E0_y)
            self.e = np.abs(E1_x)**2 + np.abs(E1_y)**2

    def update_polarization_matrices(self, L0: np.ndarray, M0: np.ndarray,
                                     N0: np.ndarray, jones_matrix: np.ndarray):
        """Update polarization matrices after interaction with surface."""
        # merge k-vector components into matrix for speed
        k0 = np.array([L0, M0, N0]).T
        k1 = np.array([self.L, self.M, self.N]).T

        # find s-component
        s = np.cross(k0, k1)
        mag = np.linalg.norm(s, axis=1)

        # handle case when mag = 0 (i.e., k0 parallel to k1)
        if np.any(mag == 0):
            s[mag == 0] = np.cross(k0[mag == 0], np.array([1, 1e-10, 0]))
            mag = np.linalg.norm(s, axis=1)

        s /= mag[:, np.newaxis]

        # find p-component pre and post surface
        p0 = np.cross(k0, s)
        p1 = np.cross(k1, s)

        # othogonal transformation matrices
        o_in = np.stack((s, p0, k0), axis=1)
        o_out = np.stack((s, p1, k1), axis=2)

        # compute polarization matrix for surface
        p = np.einsum('nij,njk,nkl->nil', o_out, jones_matrix, o_in)

        # update polarization matrices of rays
        self.p = np.matmul(p, self.p)

    def _get_3d_electric_field(self, state: PolarizationState):
        """Get 3D electric fields given polarization state and current rays."""
        k = np.array([self.L, self.M, self.N]).T

        # TODO - efficiently handle case when k parallel to x-axis
        x = np.array([1.0, 0.0, 0.0])
        if np.any(np.dot(k, x)):
            raise ValueError('Ray direction vector parallel to x-axis. '
                             'Determination of s and p vectors is ambiguous.')

        p = np.cross(k, x)
        p /= np.linalg.norm(p, axis=1)[:, np.newaxis]
        s = np.cross(p, k)

        E = (state.Ex * np.exp(1j * state.phase_x) * s +
             state.Ey * np.exp(1j * state.phase_y) * p)

        return E
