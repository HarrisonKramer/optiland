"""Provides a spline interpolator for the torch backend."""

from __future__ import annotations

import torch

from optiland.backend.interpolation.base import SplineInterpolator


def _compute_spline_coeffs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the coefficients for a batch of 1D cubic splines.
    This is a direct translation of the scipy implementation for a uniform grid.
    """
    is_1d = y.dim() == 1
    if is_1d:
        y = y.unsqueeze(1)

    n, m = y.shape
    h = x[1:] - x[:-1]

    A = torch.zeros((n, n), device=x.device, dtype=x.dtype)
    B = torch.zeros((n, m), device=x.device, dtype=x.dtype)

    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

    # This is the corrected B-matrix calculation
    B[1:-1] = 3 * (y[2:] - y[:-2]) / h[1:].unsqueeze(1)

    # Solve for the derivatives at the knots
    s = torch.linalg.solve(A, B)

    # Compute the coefficients
    a = y[:-1]
    b = (y[1:] - y[:-1]) / h.unsqueeze(1) - h.unsqueeze(1) * (s[1:] + 2 * s[:-1]) / 3
    c = s[:-1]
    d = (s[1:] - s[:-1]) / (3 * h.unsqueeze(1))

    coeffs = torch.stack([a, b, c, d], dim=-1)

    if is_1d:
        coeffs = coeffs.squeeze(1)

    return coeffs


class TorchSplineInterpolator(SplineInterpolator):
    """A spline interpolator for the torch backend."""

    def __init__(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        grid: torch.Tensor,
    ):
        super().__init__(x_coords, y_coords, grid)
        self._recompute_coeffs()

    def _recompute_coeffs(self):
        # 1. For each column, compute spline coeffs along the y-axis
        coeffs_y = _compute_spline_coeffs(self.y_coords, self.grid)

        # 2. For each row of coeffs, compute spline coeffs along the x-axis
        # Reshape to treat each row of coefficients as a separate spline problem
        coeffs_y_reshaped = coeffs_y.permute(0, 2, 1).reshape(-1, len(self.x_coords))
        coeffs_x = _compute_spline_coeffs(self.x_coords, coeffs_y_reshaped.T)
        coeffs_x = (
            coeffs_x.permute(1, 0, 2)
            .reshape(-1, len(self.y_coords) - 1, 4)
            .permute(1, 0, 2)
        )

        # 3. Reshape to final coefficient tensor
        self.coeffs = coeffs_x.reshape(
            len(self.y_coords) - 1, len(self.x_coords) - 1, 4, 4
        )

    @property
    def grid(self) -> torch.Tensor:
        """The grid of values to interpolate."""
        return self._grid

    @grid.setter
    def grid(self, new_grid: torch.Tensor):
        self._grid = new_grid
        self._recompute_coeffs()

    def ev(
        self, y: torch.Tensor, x: torch.Tensor, dx: int = 0, dy: int = 0
    ) -> torch.Tensor:
        """Evaluates the spline at the given coordinates."""
        x_indices = torch.searchsorted(self.x_coords, x, right=True) - 1
        y_indices = torch.searchsorted(self.y_coords, y, right=True) - 1

        x_indices = torch.clamp(x_indices, 0, len(self.x_coords) - 2)
        y_indices = torch.clamp(y_indices, 0, len(self.y_coords) - 2)

        x_rel = x - self.x_coords[x_indices]
        y_rel = y - self.y_coords[y_indices]

        coeffs_grid = self.coeffs[y_indices, x_indices]

        # Powers of x_rel and y_rel
        x_pows = torch.stack([x_rel**i for i in range(4)], dim=-1)
        y_pows = torch.stack([y_rel**i for i in range(4)], dim=-1)

        if dx == 1:
            x_pows = torch.stack(
                [
                    torch.zeros_like(x_rel),
                    torch.ones_like(x_rel),
                    2 * x_rel,
                    3 * x_rel**2,
                ],
                dim=-1,
            )
        if dy == 1:
            y_pows = torch.stack(
                [
                    torch.zeros_like(y_rel),
                    torch.ones_like(y_rel),
                    2 * y_rel,
                    3 * y_rel**2,
                ],
                dim=-1,
            )

        y_eval = torch.einsum("...i,...ij->...j", y_pows, coeffs_grid)
        result = torch.einsum("...i,...i->...", y_eval, x_pows)

        return result
