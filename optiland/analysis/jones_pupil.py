"""Jones Pupil Analysis

This module provides a Jones pupil analysis for optical systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.analysis.base import BaseAnalysis
from optiland.rays import PolarizationState
from optiland.utils import resolve_fields

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from optiland.optic import Optic


class JonesPupil(BaseAnalysis):
    """Generates and plots Jones pupil maps.

    This class computes the spatially resolved Jones matrix at the exit pupil
    (or image plane) as a function of normalized pupil coordinates. It visualizes
    the real and imaginary parts of the Jones matrix elements (Jxx, Jxy, Jyx, Jyy).

    Attributes:
        optic: Instance of the optic object to be assessed.
        fields: Fields at which data is generated.
        wavelengths: Wavelengths at which data is generated.
        grid_size: The side length of the square grid of rays (NxN).
        data: Contains Jones matrix data in a nested list, ordered by field, then
            wavelength.
    """

    def __init__(
        self,
        optic: Optic,
        fields: str | list = "all",
        wavelengths: str | list = "all",
        grid_size: int = 33,
    ):
        """Initializes the JonesPupil analysis.

        Args:
            optic: An instance of the optic object to be assessed.
            fields: Fields at which to generate data. If 'all', all defined
                field points are used. Defaults to "all".
            wavelengths: Wavelengths at which to generate data. If 'all', all
                defined wavelengths are used. Defaults to "all".
            grid_size: The number of points along one dimension of the pupil grid.
                Defaults to 33.
        """
        self.fields = resolve_fields(optic, fields)
        self.grid_size = grid_size
        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (16, 8),
    ) -> tuple[Figure, list[Axes]]:
        """Displays the Jones pupil plots.

        Args:
            fig_to_plot_on: An existing Matplotlib figure to plot on. If None,
                a new figure is created. Defaults to None.
            figsize: The figure size for the output window. Defaults to (16, 8).

        Returns:
            A tuple containing the Matplotlib figure and a list of its axes.
        """
        if not self.fields or self.fields == [(0, 0)]:
            # resolve_fields returns [(0,0)] if max_field is 0 which happens when
            # fields are empty or explicitly (0,0).
            # We should check if optic has fields.
            if not self.optic.fields.fields:
                return None, None

        # Determine layout
        num_fields = len(self.fields)
        num_wavelengths = len(self.wavelengths)

        # For simplicity, we plot the first field and first wavelength by default
        # if multiple are present, or we could tile them.
        # Given standard usage, usually one looks at one field/wavelength or
        # tabs through them. But BaseAnalysis `view` usually static.
        # Let's plot the first field and first wavelength for now,
        # or iterate? SpotDiagram plots all fields.
        # Jones Pupil is 8 plots per field/wavelength. That's too many to tile.
        # We will plot the PRIMARY field (or first) and PRIMARY wavelength (or first).

        # Select first field and primary wavelength index
        field_idx = 0
        wl_idx = 0
        if self.optic.primary_wavelength in self.wavelengths:
            wl_idx = self.wavelengths.index(self.optic.primary_wavelength)

        data_fw = self.data[field_idx][wl_idx]

        if fig_to_plot_on:
            fig = fig_to_plot_on
            fig.clear()
        else:
            fig = plt.figure(figsize=figsize)

        # 2 rows (Real, Imag), 4 columns (Jxx, Jxy, Jyx, Jyy)
        axs = fig.subplots(2, 4, sharex=True, sharey=True)

        # Elements to plot
        elements = [
            ("Jxx", data_fw["J"][:, 0, 0]),
            ("Jxy", data_fw["J"][:, 0, 1]),
            ("Jyx", data_fw["J"][:, 1, 0]),
            ("Jyy", data_fw["J"][:, 1, 1]),
        ]

        px = be.to_numpy(data_fw["Px"]).reshape(self.grid_size, self.grid_size)
        py = be.to_numpy(data_fw["Py"]).reshape(self.grid_size, self.grid_size)
        mask = px**2 + py**2 <= 1.0
        # px[~mask] = np.nan
        # py[~mask] = np.nan

        # pcolormesh handles NaN in C (value) array by masking the cell,
        # but X and Y coordinates must be valid.
        # We should NOT mask px and py, only the values.

        for col, (name, values) in enumerate(elements):
            val_np = be.to_numpy(values).reshape(self.grid_size, self.grid_size)
            val_np[~mask] = np.nan

            # Real part
            ax_real = axs[0, col]
            im_real = ax_real.pcolormesh(
                px, py, np.real(val_np), shading="nearest", cmap="viridis"
            )
            ax_real.set_title(f"Re({name})")
            fig.colorbar(im_real, ax=ax_real)

            # Imag part
            ax_imag = axs[1, col]
            im_imag = ax_imag.pcolormesh(
                px, py, np.imag(val_np), shading="nearest", cmap="viridis"
            )
            ax_imag.set_title(f"Im({name})")
            fig.colorbar(im_imag, ax=ax_imag)

        # Labels
        for ax in axs[:, 0]:
            ax.set_ylabel("Py")
        for ax in axs[-1, :]:
            ax.set_xlabel("Px")

        field_val = self.fields[field_idx]
        wl_val = self.wavelengths[wl_idx]
        fig.suptitle(f"Jones Pupil - Field: {field_val}, Wavelength: {wl_val:.4f} µm")
        fig.tight_layout()

        return fig, fig.get_axes()

    def _generate_data(self):
        """Generates Jones matrix data for all fields and wavelengths."""
        # Generate pupil grid
        x = be.linspace(-1.0, 1.0, self.grid_size)
        y = be.linspace(-1.0, 1.0, self.grid_size)
        Px_grid, Py_grid = be.meshgrid(x, y)
        Px = Px_grid.flatten()
        Py = Py_grid.flatten()

        data = []
        for Hx, Hy in self.fields:
            field_data = []
            for wl in self.wavelengths:
                field_data.append(self._generate_single_data(Hx, Hy, Px, Py, wl))
            data.append(field_data)

        return data

    def _generate_single_data(self, Hx, Hy, Px, Py, wavelength):
        """Generates data for a single field and wavelength configuration."""

        # Handle polarization state
        original_pol = self.optic.polarization
        if original_pol == "ignore":
            # Temporarily enable polarization to get PolarizedRays
            self.optic.set_polarization(PolarizationState())

        try:
            rays = self.optic.trace_generic(
                Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wavelength
            )
        finally:
            if original_pol == "ignore":
                self.optic.set_polarization("ignore")

        if not hasattr(rays, "p"):
            # Fallback if rays are not polarized (should not happen with the check above)
             raise RuntimeError("Ray tracing did not return polarized rays.")

        # Ray direction vectors (normalized)
        k = be.stack([rays.L, rays.M, rays.N], axis=1)
        # Normalize k (should be already, but to be safe)
        k_norm = be.linalg.norm(k, axis=1)
        k = k / be.unsqueeze_last(k_norm)

        # Construct local basis vectors (Standard Polar Projection / Dipole-like)
        # v ~ Y-axis: perpendicular to k and X=[1,0,0]
        x_axis = be.array([1.0, 0.0, 0.0])
        # Broadcast x_axis to match k shape
        x_axis = be.broadcast_to(x_axis, k.shape)

        v = be.cross(k, x_axis)
        v_norm = be.linalg.norm(v, axis=1)

        # Handle singularity: if k is parallel to X (v_norm ~ 0)
        # For optical systems, k is rarely along X.
        # If it happens, we can default to Z-axis for the cross product?
        # But for now assuming paraxial-ish along Z.

        # Avoid division by zero
        v = v / be.unsqueeze_last(v_norm + 1e-15)

        # u ~ X-axis: perpendicular to v and k
        u = be.cross(v, k)
        u_norm = be.linalg.norm(u, axis=1)
        u = u / be.unsqueeze_last(u_norm + 1e-15)

        # Basis matrices
        # Input basis (Global X, Global Y) assumed for Jones Matrix definition
        # The stored 'p' matrix is Global -> Global.
        # We want J such that E_local_out = J * E_global_in
        # (Assuming input is typically defined in global coords at entrance pupil)

        # Project global P onto local basis (u, v)
        # Jxx = u . (P . x_in)
        # Jxy = u . (P . y_in)
        # Jyx = v . (P . x_in)
        # Jyy = v . (P . y_in)

        x_in = be.array([1.0, 0.0, 0.0])
        y_in = be.array([0.0, 1.0, 0.0])

        # p has shape (N, 3, 3)
        # P . x_in is simply the first column of p
        # P . y_in is simply the second column of p

        P_x_in = rays.p[:, :, 0] # Shape (N, 3)
        P_y_in = rays.p[:, :, 1] # Shape (N, 3)

        # Dot products
        Jxx = be.sum(u * P_x_in, axis=1)
        Jxy = be.sum(u * P_y_in, axis=1)
        Jyx = be.sum(v * P_x_in, axis=1)
        Jyy = be.sum(v * P_y_in, axis=1)

        # Stack into (N, 2, 2)
        row1 = be.stack([Jxx, Jxy], axis=1)
        row2 = be.stack([Jyx, Jyy], axis=1)
        J = be.stack([row1, row2], axis=1)

        return {
            "Px": Px,
            "Py": Py,
            "J": J
        }
