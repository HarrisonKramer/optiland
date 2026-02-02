"""
Reporting Engine Module.

This module contains the core logic that assembles a report.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.physical_apertures.radial import RadialAperture
from optiland.reporting.metrics import (
    FirstOrderMetrics,
    MaterialMetrics,
    SeidelMetrics,
)
from optiland.reporting.pdf_export import PDFReportRenderer
from optiland.reporting.visualizations import (
    FieldCurvatureDistortionPlot,
    MTFPlot,
    OPDPlot,
    PlotStyler,
    RayFanPlot,
    SpotDiagramPlot,
)

if TYPE_CHECKING:
    from optiland.optic.optic import Optic


class ReportBuilder(abc.ABC):
    """Abstract base class for report builders."""

    def __init__(self, optic: Optic):
        self.optic = optic

    @abc.abstractmethod
    def save(self, filename: str):
        """Builds and saves the report to a file."""
        pass


class StandardPerformanceReport(ReportBuilder):
    """Generates a standard performance report."""

    def save(self, filename: str):
        PlotStyler.apply_style()

        with PDFReportRenderer(filename) as pdf:
            # 1. Dashboard / Title Page
            self._create_dashboard(pdf)

            # 2. System Prescription
            self._create_prescription_table(pdf)

            # 3. First Order Properties (Detailed)
            self._create_first_order_table(pdf)

            # 4. Image Quality Assessment
            self._create_image_quality_section(pdf)

            # 5. Aberration Analysis
            self._create_aberration_analysis(pdf)

    def _create_dashboard(self, pdf: PDFReportRenderer):
        """Creates the dashboard page."""
        import os

        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        from optiland.visualization.system.optic_viewer import OpticViewer

        fig = plt.figure(figsize=(8.5, 11))

        # 1. Logo and Title
        ax_header = fig.add_axes([0.05, 0.85, 0.9, 0.1])
        ax_header.axis("off")

        # Try to find logo
        logo_path = None
        possible_paths = [
            "optiland_gui/resources/logo.png",
            "../optiland_gui/resources/logo.png",
            "../../optiland_gui/resources/logo.png",
            os.path.join(
                os.path.dirname(__file__), "../../optiland_gui/resources/logo.png"
            ),
        ]

        for p in possible_paths:
            if os.path.exists(p):
                logo_path = p
                break

        if logo_path:
            try:
                img = plt.imread(logo_path)
                imagebox = OffsetImage(img, zoom=0.15)
                ab = AnnotationBbox(
                    imagebox, (0.05, 0.5), frameon=False, boxcoords="axes fraction"
                )
                ax_header.add_artist(ab)
            except Exception:
                pass  # Fail silently on logo

        # Title
        ax_header.text(
            0.5,
            0.5,
            "Optical System Performance Report",
            ha="center",
            va="center",
            fontsize=24,
            weight="bold",
            color="#2c3e50",
        )

        # 2. Lens Layout (Center)
        ax_layout = fig.add_axes([0.1, 0.45, 0.8, 0.35])

        # Use OpticViewer directly to plot on specific axes
        viewer = OpticViewer(self.optic)
        viewer.view(projection="YZ", ax=ax_layout)
        ax_layout.set_title("System Layout", fontsize=14, weight="bold")

        # 3. Summary Table (Bottom)
        fo_metrics = FirstOrderMetrics.calculate(self.optic)

        summary_data = [
            ["EFL", f"{fo_metrics['EFL']:.4f} mm"],
            ["F/#", f"{fo_metrics['F/#']:.4f}"],
            ["EPD", f"{fo_metrics['EPD']:.4f} mm"],
            ["Field of View", self._get_fov_string()],
            ["Total Track", f"{fo_metrics['Total Track']:.4f} mm"],
            [
                "Wavelengths",
                f"{self.optic.wavelengths.primary_wavelength.value:.4f} µm (Primary)",
            ],
        ]

        ax_table = fig.add_axes([0.15, 0.1, 0.7, 0.25])
        ax_table.axis("off")
        ax_table.set_title("Performance Summary", fontsize=12, weight="bold", pad=10)

        table = ax_table.table(
            cellText=summary_data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Style summary table slightly
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor("#d0d0d0")
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#2c3e50")
            else:
                cell.set_facecolor("#fcfcfc")

        pdf.add_figure(fig, add_header_footer=True)
        plt.close(fig)

    def _get_fov_string(self) -> str:
        fields = self.optic.fields
        if hasattr(fields, "max_field"):
            return f"{fields.max_field:.2f}"
        return "N/A"

    def _create_prescription_table(self, pdf: PDFReportRenderer):
        """Creates the system prescription table."""
        mat_metrics = MaterialMetrics.calculate(self.optic)

        surfaces = self.optic.surface_group.surfaces
        data = []
        for i, surf in enumerate(surfaces):
            # Radius
            radius = surf.geometry.radius
            rad_str = "Infinity" if be.isinf(radius) else f"{be.to_numpy(radius):.4f}"

            # Thickness (distance to next surface)
            if i < len(surfaces) - 1:
                th = (
                    self.optic.surface_group.positions[i + 1]
                    - self.optic.surface_group.positions[i]
                )
                th = be.to_numpy(th)
                if np.size(th) > 1:
                    th = th[0]  # Assume on-axis
            else:
                th = 0.0

            # Material from metrics
            mat_info = mat_metrics[i]
            mat_str = mat_info["Material"]
            if mat_str == "Air":
                pass

            semi_dia = "Auto"
            if surf.aperture:
                if isinstance(surf.aperture, RadialAperture):
                    semi_dia = f"{float(surf.aperture.r_max):.4f}"
                elif hasattr(surf.aperture, "max_radius"):
                    semi_dia = f"{float(surf.aperture.max_radius):.4f}"
                else:
                    try:
                        extent = surf.aperture.extent
                        max_ext = max(abs(x) for x in extent)
                        semi_dia = f"{float(max_ext):.4f}"
                    except Exception:
                        pass

            data.append(
                [
                    i,
                    surf.surface_type,
                    rad_str,
                    f"{float(th):.4f}",
                    mat_str,
                    semi_dia,
                ]
            )

        cols = [
            "Surf",
            "Type",
            "Radius",
            "Thickness",
            "Material",
            "Semi-Diameter",
        ]
        fig = pdf.render_table(data, cols, title="Lens Data Editor")
        pdf.add_figure(fig, add_header_footer=True)
        plt.close(fig)

    def _create_first_order_table(self, pdf: PDFReportRenderer):
        metrics = FirstOrderMetrics.calculate(self.optic)
        data = [[k, f"{v:.4f}"] for k, v in metrics.items()]
        fig = pdf.render_table(
            data, ["Metric", "Value"], title="First Order Properties"
        )
        pdf.add_figure(fig, add_header_footer=True)
        plt.close(fig)

    def _create_image_quality_section(self, pdf: PDFReportRenderer):
        # 1. Spot Diagram
        spot_plot = SpotDiagramPlot(self.optic)
        fig = spot_plot.plot()
        fig.suptitle("Spot Diagram", fontsize=16, weight="bold")
        pdf.add_figure(fig, add_header_footer=True)
        plt.close(fig)

        # 2. MTF
        mtf_plot = MTFPlot(self.optic)
        fig = mtf_plot.plot()
        fig.suptitle("Geometric MTF", fontsize=16, weight="bold")
        pdf.add_figure(fig, add_header_footer=True)
        plt.close(fig)

        # 3. Field Curvature & Distortion
        fcd_plot = FieldCurvatureDistortionPlot(self.optic)
        fig = fcd_plot.plot()
        fig.suptitle("Field Curvature & Distortion", fontsize=16, weight="bold")
        pdf.add_figure(fig, add_header_footer=True)
        plt.close(fig)

        # 4. Ray Fans
        rf_plot = RayFanPlot(self.optic)
        fig = rf_plot.plot()
        fig.suptitle("Ray Fan", fontsize=16, weight="bold")
        pdf.add_figure(fig, add_header_footer=True)
        plt.close(fig)

        # 5. OPD Fans
        opd_plot = OPDPlot(self.optic)
        fig = opd_plot.plot()
        fig.suptitle("OPD Fan", fontsize=16, weight="bold")
        pdf.add_figure(fig, add_header_footer=True)
        plt.close(fig)

    def _create_aberration_analysis(self, pdf: PDFReportRenderer):
        metrics = SeidelMetrics.calculate(self.optic)
        data = [[k, f"{v:.4f}"] for k, v in metrics.items()]
        fig = pdf.render_table(
            data,
            ["Aberration", "Coefficient (Seidel)"],
            title="Third-Order Aberrations",
        )
        pdf.add_figure(fig, add_header_footer=True)
        plt.close(fig)
