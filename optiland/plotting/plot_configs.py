"""Plotting configuration objects."""

from typing import Optional, TypedDict  # Removed Tuple


class LegendConfig(TypedDict, total=False):
    """Configuration for plot legends.

    Attributes:
        show_legend: Whether to display the legend.
        legend_loc: The location of the legend.
        legend_title: The title of the legend.
        legend_frameon: Whether to draw a frame around the legend.
        legend_shadow: Whether to draw a shadow behind the legend.
        legend_fancybox: Whether to use a fancy box for the legend.
        legend_ncol: The number of columns in the legend.
        legend_bbox_to_anchor: The bounding box to anchor the legend to.
    """

    show_legend: Optional[bool]
    legend_loc: Optional[str]
    legend_title: Optional[str]
    legend_frameon: Optional[bool]
    legend_shadow: Optional[bool]
    legend_fancybox: Optional[bool]
    legend_ncol: Optional[int]
    legend_bbox_to_anchor: Optional[tuple[float, float]]
