"""
PDF Export Module for Optiland Reporting.

This module provides a rendering engine to compile reports into a single,
clean paginated PDF using matplotlib's backend_pdf.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class PDFReportRenderer:
    """Renders reports to PDF."""

    def __init__(self, filename: str):
        self.filename = filename
        self.pdf = PdfPages(filename)

    def add_figure(self, fig: Figure):
        """Adds a matplotlib figure to the PDF."""
        self.pdf.savefig(fig, bbox_inches="tight")

    def render_table(
        self, data: list[list[Any]], col_labels: list[str], title: str = ""
    ) -> Figure:
        """Renders a data table as a matplotlib figure.

        Args:
            data: 2D list of data values.
            col_labels: List of column headers.
            title: Table title.

        Returns:
            A Figure object containing the rendered table.
        """
        fig, ax = plt.subplots(figsize=(8.5, 11))  # Portrait
        ax.axis("tight")
        ax.axis("off")

        if title:
            ax.set_title(title, fontsize=14, weight="bold", pad=20)

        table = ax.table(
            cellText=data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # More vertical padding

        return fig

    def close(self):
        """Closes the PDF file."""
        self.pdf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
