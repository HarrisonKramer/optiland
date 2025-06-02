"""Real Rays

This module contains the RealRays class, which represents a collection of real
rays in 3D space.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.materials import BaseMaterial
from optiland.rays.base import BaseRays


class RealRays(BaseRays):
    """Represents a collection of real rays in 3D space.

    This class stores the spatial coordinates (x, y, z), direction cosines
    (L, M, N), intensity, wavelength, and optical path difference for a set of
    rays. It provides methods for ray manipulation like propagation, rotation,
    refraction, and reflection.

    Attributes:
        x (be.Tensor): The x-coordinates of the rays.
        y (be.Tensor): The y-coordinates of the rays.
        z (be.Tensor): The z-coordinates of the rays.
        L (be.Tensor): The x-components of the direction vectors of the rays.
        M (be.Tensor): The y-components of the direction vectors of the rays.
        N (be.Tensor): The z-components of the direction vectors of the rays.
        i (be.Tensor): The intensity of each ray.
        w (be.Tensor): The wavelength of each ray (in micrometers).
        opd (be.Tensor): The optical path difference accumulated by each ray.
        L0 (Optional[be.Tensor]): The x-components of the direction vectors
            before the last interaction (e.g., refraction/reflection). Used for
            calculations like aligning surface normals.
        M0 (Optional[be.Tensor]): The y-components of the direction vectors
            before the last interaction.
        N0 (Optional[be.Tensor]): The z-components of the direction vectors
            before the last interaction.
        is_normalized (bool): A flag indicating whether the direction vectors
            (L, M, N) are currently normalized. Set to False when operations
            might denormalize them, and True after normalization.
    """

    def __init__(self, x, y, z, L, M, N, intensity, wavelength):
        """Initializes a RealRays object.

        Args:
            x (float | list[float] | be.Tensor): The initial x-coordinates.
            y (float | list[float] | be.Tensor): The initial y-coordinates.
            z (float | list[float] | be.Tensor): The initial z-coordinates.
            L (float | list[float] | be.Tensor): The initial x-components of the
                direction vectors.
            M (float | list[float] | be.Tensor): The initial y-components of the
                direction vectors.
            N (float | list[float] | be.Tensor): The initial z-components of the
                direction vectors.
            intensity (float | list[float] | be.Tensor): The initial intensity
                of the rays.
            wavelength (float | list[float] | be.Tensor): The wavelength of each
                ray in micrometers.
        """
        self.x = be.as_array_1d(x)
        self.y = be.as_array_1d(y)
        self.z = be.as_array_1d(z)
        self.L = be.as_array_1d(L)
        self.M = be.as_array_1d(M)
        self.N = be.as_array_1d(N)
        self.i = be.as_array_1d(intensity)
        self.w = be.as_array_1d(wavelength)
        self.opd = be.zeros_like(self.x)

        # variables to hold pre-surface direction cosines
        self.L0 = None
        self.M0 = None
        self.N0 = None

        self.is_normalized = True

    def rotate_x(self, rx: float):
        """Rotates the ray coordinates and direction vectors about the x-axis.

        Args:
            rx (float): The rotation angle in radians.
        """
        rx = be.array(rx)
        y = self.y * be.cos(rx) - self.z * be.sin(rx)
        z = self.y * be.sin(rx) + self.z * be.cos(rx)
        m = self.M * be.cos(rx) - self.N * be.sin(rx)
        n = self.M * be.sin(rx) + self.N * be.cos(rx)
        self.y = y
        self.z = z
        self.M = m
        self.N = n

    def rotate_y(self, ry: float):
        """Rotates the ray coordinates and direction vectors about the y-axis.

        Args:
            ry (float): The rotation angle in radians.
        """
        ry = be.array(ry)
        x = self.x * be.cos(ry) + self.z * be.sin(ry)
        z = -self.x * be.sin(ry) + self.z * be.cos(ry)
        L = self.L * be.cos(ry) + self.N * be.sin(ry)
        n = -self.L * be.sin(ry) + self.N * be.cos(ry)
        self.x = x
        self.z = z
        self.L = L
        self.N = n

    def rotate_z(self, rz: float):
        """Rotates the ray coordinates and direction vectors about the z-axis.

        Args:
            rz (float): The rotation angle in radians.
        """
        rz = be.array(rz)
        x = self.x * be.cos(rz) - self.y * be.sin(rz)
        y = self.x * be.sin(rz) + self.y * be.cos(rz)
        L = self.L * be.cos(rz) - self.M * be.sin(rz)
        m = self.L * be.sin(rz) + self.M * be.cos(rz)
        self.x = x
        self.y = y
        self.L = L
        self.M = m

    def propagate(self, t: float, material: Optional[BaseMaterial] = None):
        """Propagates the rays a distance `t` through a medium.

        Updates ray positions based on their direction vectors and distance `t`.
        If `material` is provided, absorption is calculated and intensity `i`
        is updated. Optical path difference `opd` is also updated if `material`
        is provided (assuming `opd` is handled by the material interaction,
        otherwise this should be added).

        Args:
            t (float): The distance to propagate the rays.
            material (Optional[BaseMaterial]): The material through which the
                rays are propagating. If provided, absorption will be applied
                to ray intensities. Defaults to None (no absorption).
        """
        self.x = self.x + t * self.L
        self.y = self.y + t * self.M
        self.z = self.z + t * self.N

        if material is not None:
            k = material.k(self.w) # Extinction coefficient for absorption
            alpha = 4 * be.pi * k / self.w  # Absorption coefficient
            # Convert t from optical system units (e.g. mm) to microns for alpha
            # This assumes self.w is in microns.
            # Ensure consistent unit handling in practice.
            self.i = self.i * be.exp(-alpha * t * 1e3)

        # normalize, if required
        if not self.is_normalized:
            self.normalize()

    def clip(self, condition: be.Tensor | bool):
        """Clips rays based on a boolean condition.

        Sets the intensity `i` of rays to zero where the condition is True.

        Args:
            condition (be.Tensor | bool): A boolean tensor or scalar. Rays where
                the condition is True will be clipped (intensity set to 0).
        """
        cond = be.array(condition)
        try:
            cond = cond.astype(bool)
        except AttributeError:
            cond = cond.bool()
        self.i = be.where(cond, be.zeros_like(self.i), self.i)

    def refract(
        self,
        nx: be.Tensor,
        ny: be.Tensor,
        nz: be.Tensor,
        n1: float | be.Tensor,
        n2: float | be.Tensor,
    ):
        """Refracts rays at an interface between two media.

        Updates the ray direction vectors (L, M, N) based on Snell's law.
        The surface normal (nx, ny, nz) is assumed to be pointing from medium 1
        to medium 2. Stores current L, M, N in L0, M0, N0.

        Args:
            nx (be.Tensor): The x-component of the surface normal vectors.
            ny (be.Tensor): The y-component of the surface normal vectors.
            nz (be.Tensor): The z-component of the surface normal vectors.
            n1 (float | be.Tensor): Refractive index of the medium the ray is
                currently in (incident medium).
            n2 (float | be.Tensor): Refractive index of the medium the ray is
                entering (transmitting/refracting medium).
        """
        # Store current direction cosines for use in calculations and by other methods
        self.L0 = be.copy(self.L)
        self.M0 = be.copy(self.M)
        self.N0 = be.copy(self.N)

        u = n1 / n2  # Ratio of refractive indices (eta_incident / eta_transmitted)

        # Align surface normal with incident ray (convention for vector formula)
        # dot is cos(theta_1), where theta_1 is angle between incident ray and normal
        nx_aligned, ny_aligned, nz_aligned, dot_incident_normal = self._align_surface_normal(nx, ny, nz)

        # Calculate the term under the square root for Snell's law vector form
        # This term is related to cos(theta_2)^2, where theta_2 is refracted angle.
        # If term_under_sqrt < 0, it indicates Total Internal Reflection (TIR).
        # be.sqrt will produce NaN for negative inputs if not handled by backend.
        term_under_sqrt = 1 - u**2 * (1 - dot_incident_normal**2)
        root = be.sqrt(term_under_sqrt) # cos(theta_2) scaled by a factor

        # Vector form of Snell's Law:
        # K_transmitted = u * K_incident + (root - u * dot_incident_normal) * Normal_aligned
        tx = u * self.L0 + (root - u * dot_incident_normal) * nx_aligned
        ty = u * self.M0 + (root - u * dot_incident_normal) * ny_aligned
        tz = u * self.N0 + (root - u * dot_incident_normal) * nz_aligned

        self.L = tx
        self.M = ty
        self.N = tz

    def reflect(self, nx: be.Tensor, ny: be.Tensor, nz: be.Tensor):
        """Reflects rays at a surface.

        Updates the ray direction vectors (L, M, N) according to the law of
        reflection. Stores current L, M, N in L0, M0, N0.

        Args:
            nx (be.Tensor): The x-component of the surface normal vectors.
            ny (be.Tensor): The y-component of the surface normal vectors.
            nz (be.Tensor): The z-component of the surface normal vectors.
        """
        self.L0 = be.copy(self.L)
        self.M0 = be.copy(self.M)
        self.N0 = be.copy(self.N)

        # Align surface normal with incident ray
        # dot is cos(theta_incident), angle between incident ray and normal
        nx_aligned, ny_aligned, nz_aligned, dot_incident_normal = self._align_surface_normal(nx, ny, nz)

        # Vector form of the Law of Reflection:
        # K_reflected = K_incident - 2 * dot_incident_normal * Normal_aligned
        self.L = self.L0 - 2 * dot_incident_normal * nx_aligned
        self.M = self.M0 - 2 * dot_incident_normal * ny_aligned
        self.N = self.N0 - 2 * dot_incident_normal * nz_aligned

    def update(self, jones_matrix: Optional[be.Tensor] = None):
        """Update ray properties after interaction with a surface.

        This method is intended to be overridden by subclasses that require
        updating specific ray properties (e.g., polarization). The base
        implementation in `RealRays` does nothing.

        Args:
            jones_matrix (be.Tensor, optional): An optional Jones matrix or
                similar parameter that might be used by subclasses.
                Defaults to None.
        """
        pass

    def normalize(self):
        """Normalize the direction vectors of the rays."""
        mag = be.sqrt(self.L**2 + self.M**2 + self.N**2)
        self.L = self.L / mag
        self.M = self.M / mag
        self.N = self.N / mag
        self.is_normalized = True

    def _align_surface_normal(
        self, nx: be.Tensor, ny: be.Tensor, nz: be.Tensor
    ) -> tuple[be.Tensor, be.Tensor, be.Tensor, be.Tensor]:
        """Aligns the surface normal with the incident ray vectors.

        This convention ensures the surface normal points towards the incident
        ray side, which is required for consistent vector reflection and
        refraction calculations. The normal is flipped if it points away from
        the incident ray.

        Args:
            nx (be.Tensor): The x-component of the surface normal vector(s).
            ny (be.Tensor): The y-component of the surface normal vector(s).
            nz (be.Tensor): The z-component of the surface normal vector(s).

        Returns:
            tuple[be.Tensor, be.Tensor, be.Tensor, be.Tensor]:
                - nx: The aligned x-component of the surface normal.
                - ny: The aligned y-component of the surface normal.
                - nz: The aligned z-component of the surface normal.
                - dot: The absolute dot product of the original surface normal
                  and the incident ray direction vectors (L0, M0, N0).
        """
        dot = self.L0 * nx + self.M0 * ny + self.N0 * nz

        sgn = be.sign(dot)
        nx = nx * sgn
        ny = ny * sgn
        nz = nz * sgn

        dot = be.abs(dot)
        return nx, ny, nz, dot

    def __str__(self):
        """Returns a string representation of the rays in a tabular format.
        Truncates output if the number of rays is large, showing first,
        central, and last rays.
        """

        if self.x is None or len(self.x) == 0:
            return "RealRays object (No rays)"

        num_rays = len(self.x)
        max_rays_to_print = 3
        header = (
            f"{'Ray #':>6} | {'x':>10} | {'y':>10} | {'z':>10} | "
            f"{'L':>10} | {'M':>10} | {'N':>10} | "
            f"{'Intensity':>10} | {'Wavelength':>12}\n"
        )
        separator = "-" * (len(header) + 5) + "\n"

        table = header + separator

        def format_ray(i):
            if 0 <= i < num_rays:
                x = be.to_numpy(self.x)[i]
                y = be.to_numpy(self.y)[i]
                z = be.to_numpy(self.z)[i]
                L = be.to_numpy(self.L)[i]
                M = be.to_numpy(self.M)[i]
                N = be.to_numpy(self.N)[i]
                intensity = be.to_numpy(self.i)[i]
                wavelength = be.to_numpy(self.w)[i]
                txt = (
                    f"{i:6} | {x:10.4f} | {y:10.4f} | {z:10.4f} | {L:10.6f} | "
                    f"{M:10.6f} | {N:10.6f} | "
                    f"{intensity:10.4f} | {wavelength:12.4f}\n"
                )
                return txt
            return ""

        if num_rays <= max_rays_to_print:
            indices_to_print = list(range(num_rays))
            count_shown = num_rays

            for i in indices_to_print:
                table += format_ray(i)

        else:
            num_ends = (max_rays_to_print - 1) // 2
            central_index = num_rays // 2

            indices = (
                set(range(num_ends))
                | {central_index}
                | set(range(num_rays - num_ends, num_rays))
            )

            sorted_indices = sorted(list(indices))
            count_shown = len(sorted_indices)

            for i in sorted_indices:
                table += format_ray(i)

        table += separator
        table += f"Showing {count_shown} of {num_rays} rays.\n"

        return table
