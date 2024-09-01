"""Optiland Rays Module

This module defines classes for representing and manipulating rays in optical
simulations. It includes functionality for translating and rotating rays in
three-dimensional space, as well as initializing rays with specific properties
such as position, direction, intensity, and wavelength.

Kramer Harrison, 2024
"""
from typing import Optional
import numpy as np
from optiland.materials import BaseMaterial


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
        elif isinstance(data, list):
            return np.array(data, dtype=float)
        elif isinstance(data, np.ndarray):
            return np.ravel(data).astype(float)
        else:
            raise ValueError('Unsupported input type. Must be a scalar, '
                             'a list, or a NumPy array.')


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
        i (ndarray): The intensity of the rays.
        w (ndarray): The wavelength of the rays.
        opd (ndarray): The optical path length of the rays.

    Methods:
        rotate_x(rx: float): Rotate the rays about the x-axis.
        rotate_y(ry: float): Rotate the rays about the y-axis.
        rotate_z(rz: float): Rotate the rays about the z-axis.
        propagate(t: float): Propagate the rays a distance t.
        clip(condition): Clip the rays based on a condition.
    """

    def __init__(self, x, y, z, L, M, N, intensity, wavelength):
        self.x = self._process_input(x)
        self.y = self._process_input(y)
        self.z = self._process_input(z)
        self.L = self._process_input(L)
        self.M = self._process_input(M)
        self.N = self._process_input(N)
        self.i = self._process_input(intensity)
        self.w = self._process_input(wavelength)
        self.opd = np.zeros_like(self.x)

        # variables to hold pre-surface direction cosines
        self.L0 = None
        self.M0 = None
        self.N0 = None

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

    def propagate(self, t: float, material: BaseMaterial = None):
        """Propagate the rays a distance t."""
        self.x += t * self.L
        self.y += t * self.M
        self.z += t * self.N

        if material is not None:
            k = material.k(self.w)
            alpha = 4 * np.pi * k / self.w
            self.i *= np.exp(-alpha * t * 1e3)  # mm to microns

    def clip(self, condition):
        """Clip the rays based on a condition."""
        self.i[condition] = 0.0

    def refract(self, nx, ny, nz, n1, n2):
        """
        Refract rays on the surface.

        Args:
            rays: The rays.
            nx: The x-component of the surface normals.
            ny: The y-component of the surface normals.
            nz: The z-component of the surface normals.

        Returns:
            RealRays: The refracted rays.
        """
        self.L0 = self.L.copy()
        self.M0 = self.M.copy()
        self.N0 = self.N.copy()

        u = n1 / n2
        ni = nx*self.L0 + ny*self.M0 + nz*self.N0
        root = np.sqrt(1 - u**2 * (1 - ni**2))
        tx = u * self.L0 + nx * root - u * nx * ni
        ty = u * self.M0 + ny * root - u * ny * ni
        tz = u * self.N0 + nz * root - u * nz * ni

        self.L = tx
        self.M = ty
        self.N = tz

    def reflect(self, nx, ny, nz):
        """
        Reflects the rays on the surface.

        Args:
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.

        Returns:
            RealRays: The reflected rays.
        """
        self.L0 = self.L.copy()
        self.M0 = self.M.copy()
        self.N0 = self.N.copy()

        dot = self.L * nx + self.M * ny + self.N * nz
        self.L -= 2 * dot * nx
        self.M -= 2 * dot * ny
        self.N -= 2 * dot * nz

    def update(self, jones_matrix: np.ndarray = None):
        """Update ray properties (primarily used for polarization)."""
        pass


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
        self.i = np.ones_like(self.y)
        self.w = self._process_input(wavelength)

    def propagate(self, t: float):
        """
        Propagates the rays by a given distance.

        Args:
            t (float): The distance to propagate the rays.
        """
        self.z += t
        self.y += t * self.u


class PolarizationState:
    """
    Represents the polarization state of a light ray.

    Attributes:
        is_polarized (bool): Indicates whether the state is polarized.
        Ex (Optional[float]): Electric field component in the x-direction.
        Ey (Optional[float]): Electric field component in the y-direction.
        phase_x (Optional[float]): Phase of the x-component of the electric
            field.
        phase_y (Optional[float]): Phase of the y-component of the electric
            field.
    """
    def __init__(self, is_polarized: bool = False,
                 Ex: Optional[float] = None,
                 Ey: Optional[float] = None,
                 phase_x: Optional[float] = None,
                 phase_y: Optional[float] = None):
        if is_polarized:
            if None in [Ex, Ey, phase_x, phase_y]:
                raise ValueError('All parameters must be provided for a '
                                 'polarized state.')
        else:
            if not all(var is None for var in [Ex, Ey, phase_x, phase_y]):
                raise ValueError('Ex, Ey, phase_x, and phase_y must be None '
                                 'for a non-polarized state.')

        self.is_polarized = is_polarized
        self.Ex = float(Ex) if Ex is not None else None
        self.Ey = float(Ey) if Ey is not None else None
        self.phase_x = float(phase_x) if phase_x is not None else None
        self.phase_y = float(phase_y) if phase_y is not None else None

        if self.Ex is not None and self.Ey is not None:
            mag = np.sqrt(self.Ex**2 + self.Ey**2)
            self.Ex /= mag
            self.Ey /= mag

    def __str__(self):
        """
        Returns a string representation of the polarization state.

        Returns:
            str: The string representation of the polarization state.
        """
        if self.is_polarized:
            return f"Polarized Light: Ex: {self.Ex}, Ey: {self.Ey}, " \
                   f"Phase x: {self.phase_x}, Phase y: {self.phase_y}"
        else:
            return "Unpolarized Light"

    def __repr__(self):
        """
        Returns a string representation of the polarization state.

        Returns:
            str: The string representation of the polarization state.
        """
        return self.__str__()


def create_polarization(pol_type: str):
    """
    Create a polarization state based on the given polarization type.

    Args:
        pol_type (str): The type of polarization. Must be one of the following:
            - 'unpolarized' for unpolarized light
            - 'H' for horizontal polarization
            - 'V' for vertical polarization
            - 'L+45' for linear polarization at +45 degrees
            - 'L-45' for linear polarization at -45 degrees
            - 'RCP' for right circular polarization
            - 'LCP' for left circular polarization

    Returns:
        PolarizationState: The created polarization state.

    Raises:
        ValueError: If an invalid polarization type is provided.
    """
    if pol_type == 'unpolarized':
        return PolarizationState(is_polarized=False)
    elif pol_type == 'H':
        Ex = 1
        Ey = 0
        phase_x = 0
        phase_y = 0
    elif pol_type == 'V':
        Ex = 0
        Ey = 1
        phase_x = 0
        phase_y = 0
    elif pol_type == 'L+45':
        Ex = 1
        Ey = 1
        phase_x = 0
        phase_y = 0
    elif pol_type == 'L-45':
        Ex = 1
        Ey = -1
        phase_x = 0
        phase_y = 0
    elif pol_type == 'RCP':
        Ex = np.sqrt(2) / 2
        Ey = np.sqrt(2) / 2
        phase_x = 0
        phase_y = -np.pi / 2
    elif pol_type == 'LCP':
        Ex = np.sqrt(2) / 2
        Ey = np.sqrt(2) / 2
        phase_x = 0
        phase_y = np.pi / 2
    else:
        raise ValueError('Invalid polarization type. Must be H, V, L+45, L-45,'
                         ' RCP or LCP.')
    return PolarizationState(is_polarized=True, Ex=Ex, Ey=Ey,
                             phase_x=phase_x, phase_y=phase_y)


class PolarizedRays(RealRays):
    """
    Represents a class for polarized rays in three-dimensional space.

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
        """
        Compute the output electric field given the input electric field.

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
            self.i = np.sum(np.abs(E1)**2, axis=1)
        else:
            # Local x-axis field
            state_x = PolarizationState(is_polarized=True, Ex=1.0, Ey=0.0,
                                        phase_x=0.0, phase_y=0.0)
            E0_x = self._get_3d_electric_field(state_x)
            E1_x = self.get_output_field(E0_x)

            # Local y-axis field
            state_y = PolarizationState(is_polarized=True, Ex=0.0, Ey=1.0,
                                        phase_x=0.0, phase_y=0.0)
            E0_y = self._get_3d_electric_field(state_y)
            E1_y = self.get_output_field(E0_y)

            # average two orthogonal polarizations to get mean intensity,
            # scale by initial ray intensity
            self.i = (np.sum(np.abs(E1_x)**2, axis=1) +
                      np.sum(np.abs(E1_y)**2, axis=1)) * self._i0 / 2

    def update(self, jones_matrix: np.ndarray = None):
        """
        Update polarization matrices after interaction with surface.

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
            p = np.einsum('nij,njk,nkl->nil', o_out, jones_matrix, o_in)

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
        try:
            p /= np.linalg.norm(p, axis=1)[:, np.newaxis]
        except ZeroDivisionError:
            raise ValueError('k-vector parallel to x-axis is not currently '
                             'supported.')

        s = np.cross(p, k)

        E = (state.Ex * np.exp(1j * state.phase_x) * s +
             state.Ey * np.exp(1j * state.phase_y) * p)

        return E


class RayGenerator:
    """
    Generator class for creating rays.
    """
    def __init__(self, optic):
        self.optic = optic

    def generate_rays(self, Hx, Hy, Px, Py, wavelength):
        """
        Generates rays for tracing based on the given parameters.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            x1 (float or np.ndarray): x-coordinate of the pupil point.
            y1 (float or np.ndarray): y-coordinate of the pupil point.
            z1 (float or np.ndarray): z-coordinate of the pupil point.
            wavelength (float): Wavelength of the rays.
            EPL (float): Entrance pupil position with respect to first surface.
                Default is None.

        Returns:
            RealRays: RealRays object containing the generated rays.
        """
        x0, y0, z0 = self._get_ray_origins(Hx, Hy, Px, Py)

        if self.optic.obj_space_telecentric:
            if self.optic.field_type == 'angle':
                raise ValueError('Field type cannot be "angle" for telecentric'
                                 ' object space.')
            if self.optic.aperture.ap_type == 'EPD':
                raise ValueError('Aperture type cannot be "EPD" for '
                                 'telecentric object space.')
            elif self.optic.aperture.ap_type == 'imageFNO':
                raise ValueError('Aperture type cannot be "imageFNO" for '
                                 'telecentric object space.')

            sin = self.optic.aperture.value
            z = np.sqrt(1 - sin**2) / sin + z0
            z1 = np.full_like(Px, z)
            x1 = Px + x0
            y1 = Py + y0
        else:
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            x1 = Px * EPD / 2
            y1 = Py * EPD / 2
            z1 = np.full_like(Px, EPL)

        mag = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        x0 = np.full_like(x1, x0)
        y0 = np.full_like(x1, y0)
        z0 = np.full_like(x1, z0)

        intensity = np.ones_like(x1)
        wavelength = np.ones_like(x1) * wavelength

        if self.optic.polarization == 'ignore':
            if self.optic.surface_group.uses_polarization:
                raise ValueError('Polarization must be set when surfaces have '
                                 'polarization-dependent coatings.')
            return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)
        else:
            return PolarizedRays(x0, y0, z0, L, M, N, intensity, wavelength)

    def _get_ray_origins(self, Hx, Hy, Px, Py):
        """
        Calculate the initial positions for rays originating at the object.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            x1 (float or np.ndarray): x-coordinate of the pupil point.
            y1 (float or np.ndarray): y-coordinate of the pupil point.
            EPL (float): Entrance pupil position with respect to first surface.
                Default is None.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.

        Raises:
            ValueError: If the field type is "object_height" for an object at
                infinity.

        """
        obj = self.optic.object_surface
        max_field = self.optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        if obj.is_infinite:
            if self.optic.field_type == 'object_height':
                raise ValueError('Field type cannot be "object_height" for an '
                                 'object at infinity.')
            if self.optic.obj_space_telecentric:
                raise ValueError('Object space cannot be telecentric for an '
                                 'object at infinity.')
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            # start rays just before left-most surface (1/7th of total track)
            z = self.optic.surface_group.positions[1:-1]
            offset = self.optic.total_track / 7 - np.min(z)

            # x, y, z positions of ray starting points
            x = np.tan(np.radians(field_x)) * (offset + EPL)
            y = -np.tan(np.radians(field_y)) * (offset + EPL)
            z = self.optic.surface_group.positions[1] - offset

            x0 = Px * EPD / 2 + x
            y0 = Py * EPD / 2 + y
            z0 = np.full_like(Px, z)
        else:
            if self.optic.field_type == 'object_height':
                x = field_x
                y = field_y
                z = obj.geometry.sag(x, y) + obj.geometry.cs.z

            elif self.optic.field_type == 'angle':
                EPL = self.optic.paraxial.EPL()
                z = self.optic.surface_group.positions[0]
                x = np.tan(np.radians(field_x)) * (EPL - z)
                y = -np.tan(np.radians(field_y)) * (EPL - z)

            x0 = np.full_like(Px, x)
            y0 = np.full_like(Px, y)
            z0 = np.full_like(Px, z)

        return x0, y0, z0
