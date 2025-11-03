"""Surface Plot Configuration

This module defines the SurfacePlot class, which is used to configure
and pass information about 2D plots to be rendered on surfaces in the
3D visualization.

"""

from dataclasses import dataclass, field
from typing import Any, Type


@dataclass
class SurfacePlot:
    """A dataclass to hold the configuration for a 2D plot on a surface.

    Attributes:
        surface_index (int): The index of the surface on which to project the
            plot.
        analysis_class (Type): The analysis class to be used for generating
            the plot data (e.g., SpotDiagram).
        analysis_params (dict[str, Any]): A dictionary of parameters to be
            passed to the constructor of the analysis class.

    """

    surface_index: int
    analysis_class: Type
    analysis_params: dict[str, Any] = field(default_factory=dict)
