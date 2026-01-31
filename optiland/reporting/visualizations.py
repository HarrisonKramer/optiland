"""
Visualizations Module for Optiland Reporting.

This module provides plotting engines using matplotlib, adhering to a strict
aesthetic style suitable for publication-quality reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

import optiland.backend as be
from optiland.analysis.distortion import Distortion
from optiland.analysis.field_curvature import FieldCurvature
from optiland.analysis.ray_fan import RayFan
from optiland.analysis.spot_diagram import SpotDiagram
from optiland.mtf.geometric import GeometricMTF
from optiland.wavefront.opd_fan import OPDFan

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from optiland.optic.optic import Optic


class PlotStyler:
    """Configures matplotlib for publication-quality plots."""

    @staticmethod
    def apply_style():
        """Applies the custom style settings."""
        # Reset to defaults first
        mpl.rcdefaults()

        style_params = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.figsize": (6, 4),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
        }
        rcParams.update(style_params)


class SpotDiagramPlot:
    """Generates 2D Spot Diagrams."""

    def __init__(self, optic: Optic):
        self.optic = optic

    def plot(self, fig: Figure = None) -> Figure:
        spot = SpotDiagram(self.optic)
        if fig is None:
            fig, _ = spot.view(add_airy_disk=True)
        else:
            spot.view(fig_to_plot_on=fig, add_airy_disk=True)
        return fig


class RayFanPlot:
    """Generates Ray Fan Plots."""

    def __init__(self, optic: Optic):
        self.optic = optic

    def plot(self, fig: Figure = None) -> Figure:
        fan = RayFan(self.optic)
        if fig is None:
            fig, _ = fan.view()
        else:
            fan.view(fig_to_plot_on=fig)
        return fig


class OPDPlot:
    """Generates OPD Fan Plots."""

    def __init__(self, optic: Optic):
        self.optic = optic

    def plot(self, fig: Figure = None) -> Figure:
        opd = OPDFan(self.optic)
        if fig is None:
            fig, _ = opd.view()
        else:
            opd.view(fig_to_plot_on=fig)
        return fig


class MTFPlot:
    """Generates MTF Curves."""

    def __init__(self, optic: Optic):
        self.optic = optic

    def plot(self, fig: Figure = None) -> Figure:
        mtf = GeometricMTF(self.optic)
        if fig is None:
            fig, _ = mtf.view(add_reference=True)
        else:
            mtf.view(fig_to_plot_on=fig, add_reference=True)
        return fig


class FieldCurvatureDistortionPlot:
    """Generates Field Curvature and Distortion Plots side-by-side."""

    def __init__(self, optic: Optic):
        self.optic = optic

    def plot(self, fig: Figure = None) -> Figure:
        fc = FieldCurvature(self.optic)
        dist = Distortion(self.optic)

        if fig is None:
            fig = plt.figure(figsize=(10, 4))

        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Field Curvature plotting logic
        ax1.axvline(x=0, color="k", linewidth=1, linestyle="--")
        max_val = 0
        field_norm = be.to_numpy(
            be.linspace(0, fc.optic.fields.max_field, fc.num_points)
        )

        for k, wavelength in enumerate(fc.wavelengths):
            t_curve = be.to_numpy(fc.data[k][0])
            s_curve = be.to_numpy(fc.data[k][1])

            ax1.plot(t_curve, field_norm, label=f"{wavelength:.4f} µm (T)")
            ax1.plot(
                s_curve,
                field_norm,
                linestyle="--",
                label=f"{wavelength:.4f} µm (S)",
            )

            max_val = max(
                max_val, np.max(np.abs(t_curve)), np.max(np.abs(s_curve))
            )

        ax1.set_xlabel("Field Curvature (mm)")
        ax1.set_ylabel("Field")
        if max_val > 1e-9:
            ax1.set_xlim(-max_val * 1.1, max_val * 1.1)
        ax1.grid(True)
        ax1.set_title("Field Curvature")
        # Handle legend if needed, but might be crowded. FC plot usually has legend.
        ax1.legend(fontsize=8)

        # Distortion plotting logic
        ax2.axvline(x=0, color="k", linewidth=1, linestyle="--")
        field = be.linspace(
            1e-10, dist.optic.fields.max_field, dist.num_points
        )
        field_np = be.to_numpy(field)

        for k, wavelength in enumerate(dist.wavelengths):
            dist_k_np = be.to_numpy(dist.data[k])
            ax2.plot(dist_k_np, field_np, label=f"{wavelength:.4f} µm")

        ax2.set_xlabel("Distortion (%)")
        # ax2.set_ylabel("Field") # redundant if side-by-side
        ax2.grid(True)
        ax2.set_title("Distortion")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        return fig


class LensLayoutPlot:
    """Generates 2D Lens Layout."""

    def __init__(self, optic: Optic):
        self.optic = optic

    def plot(self, fig: Figure = None) -> Figure:
        # optic.draw returns fig, ax
        # We ignore passed fig because optic.draw creates new one internally
        # unless we modify optic.draw.
        # But for report generation, we often want to create a new figure anyway.
        # If fig was passed, we might log a warning or just ignore it.
        fig_out, _ = self.optic.draw(projection="YZ")
        return fig_out
