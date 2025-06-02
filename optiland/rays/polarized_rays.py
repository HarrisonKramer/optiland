"""Polarized Rays

This module contains the `PolarizedRays` class, which represents a class for
polarized rays in three-dimensional space. The class inherits from the
`RealRays` class.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.rays.polarization_state import PolarizationState
from optiland.rays.real_rays import RealRays


class PolarizedRays(RealRays):
    """Represents a collection of polarized rays in three-dimensional space.

    This class extends `RealRays` to include polarization properties and methods
    for tracking changes in polarization state as rays interact with optical
    elements.

    Attributes:
        x (be.ndarray): The x-coordinates of the rays.
        y (be.ndarray): The y-coordinates of the rays.
        z (be.ndarray): The z-coordinates of the rays.
        L (be.ndarray): The x-components of the direction vectors of the rays.
        M (be.ndarray): The y-components of the direction vectors of the rays.
        N (be.ndarray): The z-components of the direction vectors of the rays.
        i (be.ndarray): The intensity of the rays.
        w (be.ndarray): The wavelength of the rays.
        opd (be.ndarray): The optical path length of the rays.
        p (be.ndarray): A stack of 3x3 polarization matrices, one for each ray.
            These matrices transform the electric field components.
        _i0 (be.ndarray): The initial intensity of the rays, stored for reference
            when calculating intensity for unpolarized light.
        _L0 (be.ndarray): The initial x-components of the direction vectors.
        _M0 (be.ndarray): The initial y-components of the direction vectors.
        _N0 (be.ndarray): The initial z-components of the direction vectors.
    """

    def __init__(self, x, y, z, L, M, N, intensity, wavelength):
        """Initializes a PolarizedRays object.

        Args:
            x (float | list[float] | be.ndarray): The initial x-coordinates.
            y (float | list[float] | be.ndarray): The initial y-coordinates.
            z (float | list[float] | be.ndarray): The initial z-coordinates.
            L (float | list[float] | be.ndarray): The initial x-components of the
                direction vectors.
            M (float | list[float] | be.ndarray): The initial y-components of the
                direction vectors.
            N (float | list[float] | be.ndarray): The initial z-components of the
                direction vectors.
            intensity (float | list[float] | be.ndarray): The initial intensity
                of the rays.
            wavelength (float | list[float] | be.ndarray): The wavelength of each ray.
        """
        super().__init__(x, y, z, L, M, N, intensity, wavelength)

        self.p = be.tile(be.eye(3), (be.size(self.x), 1, 1))
        self._i0 = be.copy(intensity)
        self._L0 = be.copy(L)
        self._M0 = be.copy(M)
        self._N0 = be.copy(N)

    def get_output_field(self, E: be.ndarray) -> be.ndarray:
        """Computes the output electric field after transformation by polarization matrices.

        Args:
            E (be.ndarray): The input electric field vector(s). This should be a
                Tensor of shape (num_rays, 3) or (3,) if applying to all rays.

        Returns:
            be.ndarray: The transformed electric field vector(s) of shape (num_rays, 3).
        """
        return be.mult_p_E(self.p, E)

    def update_intensity(self, state: PolarizationState):
        """Update ray intensity based on polarization state.

        Args:
            state (PolarizationState): The polarization state of the ray.

        """
        if state.is_polarized:
            E0 = self._get_3d_electric_field(state)
            E1 = self.get_output_field(E0)
            self.i = be.sum(be.abs(E1) ** 2, axis=1)
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
                (be.sum(be.abs(E1_x) ** 2, axis=1) + be.sum(be.abs(E1_y) ** 2, axis=1))
                * self._i0
                / 2
            )

    def update(self, jones_matrix=None):
        """Update polarization matrices after interaction with surface.

        Args:
            jones_matrix (Optional[be.ndarray]): Jones matrix representing the
                interaction with the surface. If not provided, the
                polarization matrix is computed assuming an identity matrix.
        """
        # merge k-vector components into matrix for speed
        k0 = be.stack([self.L0, self.M0, self.N0]).T
        k1 = be.stack([self.L, self.M, self.N]).T

        # find s-component
        s = be.cross(k0, k1)
        mag = be.linalg.norm(s, axis=1)

        # handle case when mag = 0 (i.e., k0 parallel to k1)
        mask = mag == 0
        if be.any(mask):
            fallback = be.broadcast_to(be.array([1.0, 0.0, 0.0]), k0[mask].shape)
            s[mask] = be.cross(k0[mask], fallback)
            mag = be.linalg.norm(s, axis=1)

        s = s / be.unsqueeze_last(mag)

        # find p-component pre and post surface
        p0 = be.cross(k0, s)
        p1 = be.cross(k1, s)

        # othogonal transformation matrices
        o_in = be.stack((s, p0, k0), axis=1)
        o_out = be.stack((s, p1, k1), axis=2)

        # compute polarization matrix for surface
        if jones_matrix is None:
            p = be.matmul(o_out, o_in)
        else:
            p = be.batched_chain_matmul3(o_out, jones_matrix, o_in)

        # update polarization matrices of rays
        self.p = be.matmul(p, self.p)

    def _get_3d_electric_field(self, state: PolarizationState) -> be.ndarray:
        """Get 3D electric fields given polarization state and initial rays.

        Args:
            state (PolarizationState): The polarization state of the rays.

        Returns:
            be.ndarray: The 3D electric fields, a Tensor of shape (num_rays, 3).
                The local s and p vectors defining the field are determined robustly,
                including cases where k is parallel to the x-axis.
        """
        k = be.stack([self._L0, self._M0, self._N0]).T

        # Define primary axes, broadcast to match k's shape (num_rays, 3)
        x_direction_vecs = be.broadcast_to(be.array([1.0, 0.0, 0.0]), k.shape)
        y_direction_vecs = be.broadcast_to(be.array([0.0, 1.0, 0.0]), k.shape)

        # Attempt to define p-vector by crossing k with x-axis
        p_vecs_initial = be.cross(k, x_direction_vecs)
        p_norms = be.linalg.norm(p_vecs_initial, axis=1)

        # Mask for rays where k is parallel to x-axis (p_norm is zero)
        # Add small epsilon to p_norms to avoid division by zero or issues with
        # exact zero comparisons, then check if it's effectively zero.
        # Using a fixed epsilon might be problematic; comparing p_norms directly to 0
        # should be fine if be.linalg.norm behaves well.
        # Let's assume p_norms == 0 is a reliable check for now.
        parallel_mask = p_norms < 1e-9  # Check if norm is very close to zero

        # Ensure p_norms is not zero for division to avoid NaN/Inf.
        # Where parallel_mask is True, p_norms might be zero. Use 1.0 for these cases
        # as p_vecs_initial will be zero anyway, so p_vecs_normalized_safe becomes zero.
        p_norms_safe_for_division = be.where(
            parallel_mask, be.ones_like(p_norms), p_norms
        )

        # Normalize initial p-vectors (will be zero vector if k || x, due to p_vecs_initial being zero there)
        p_vecs_normalized_safe = p_vecs_initial / be.expand_dims(
            p_norms_safe_for_division, axis=1
        )

        # For rays parallel to x-axis:
        # s-vector is chosen to be along the global y-axis.
        s_for_parallel = y_direction_vecs
        # p-vector is k x s_for_parallel. If k=[1,0,0], s=[0,1,0], then p=[0,0,1] (z-axis).
        # This needs normalization as k might not be unit or perfectly aligned.
        p_for_parallel = be.cross(k, s_for_parallel)
        p_for_parallel_norms = be.linalg.norm(p_for_parallel, axis=1)
        # Avoid division by zero if, for some edge case, k is also parallel to y_direction_vecs (e.g. k is zero vector)
        p_for_parallel_norms_safe = be.where(
            p_for_parallel_norms < 1e-9,
            be.ones_like(p_for_parallel_norms),
            p_for_parallel_norms,
        )
        p_for_parallel = p_for_parallel / be.expand_dims(
            p_for_parallel_norms_safe, axis=1
        )

        # For rays NOT parallel to x-axis:
        # p-vector is the normalized version of (k x x_axis).
        p_for_non_parallel = p_vecs_normalized_safe
        # s-vector is p_for_non_parallel x k. This also needs normalization.
        s_for_non_parallel = be.cross(p_for_non_parallel, k)
        s_for_non_parallel_norms = be.linalg.norm(s_for_non_parallel, axis=1)
        # Avoid division by zero if p_for_non_parallel is somehow parallel to k (should not happen if k is not zero)
        s_for_non_parallel_norms_safe = be.where(
            s_for_non_parallel_norms < 1e-9,
            be.ones_like(s_for_non_parallel_norms),
            s_for_non_parallel_norms,
        )
        s_for_non_parallel = s_for_non_parallel / be.expand_dims(
            s_for_non_parallel_norms_safe, axis=1
        )

        # Combine using the mask
        # Need to ensure shapes are compatible for be.where if selecting rows
        # For vector assignment, expand_dims on mask is needed
        expanded_parallel_mask = be.expand_dims(parallel_mask, axis=1)

        final_p_vecs = be.where(
            expanded_parallel_mask, p_for_parallel, p_for_non_parallel
        )
        final_s_vecs = be.where(
            expanded_parallel_mask, s_for_parallel, s_for_non_parallel
        )

        E = (
            state.Ex * be.exp(1j * state.phase_x) * final_s_vecs
            + state.Ey * be.exp(1j * state.phase_y) * final_p_vecs
        )

        return E
