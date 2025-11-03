"""
AnalysisPlot3D: A class for rendering 2D analysis plots on 3D surfaces.

This module provides the AnalysisPlot3D class, which is responsible for
generating a VTK actor that visualizes a 2D analysis plot (e.g., a spot
diagram) as a texture on a plane positioned and oriented at a specific
surface within the 3D optical system view.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import vtk

import optiland.backend as be
from vtk.util.numpy_support import numpy_to_vtk

if TYPE_CHECKING:
    from optiland.optic.optic import Optic
    from optiland.visualization.system.surface_plot import SurfacePlot


class AnalysisPlot3D:
    """A class to generate a VTK actor for a 2D analysis plot projected
    onto a 3D surface.

    This class orchestrates the process of running a given analysis,
    rasterizing its graphical output into a NumPy array, and mapping this
    array as a VTK texture onto a plane actor that matches the position,
    orientation, and size of a target optical surface.

    Args:
        optic (Optic): The optical system context.
        config (SurfacePlot): The configuration object specifying which
            analysis to run and on which surface to plot it.

    """

    def __init__(self, optic: "Optic", config: "SurfacePlot"):
        self.optic = optic
        self.config = config
        self.actor = vtk.vtkActor()
        self._create_plot_actor()

    def _get_analysis_plot_as_array(self) -> np.ndarray:
        """Runs the specified analysis and rasterizes the resulting plot into a
        NumPy array.

        Returns:
            np.ndarray: A NumPy array representing the RGB image of the plot.

        Raises:
            AttributeError: If the analysis class does not have a 'view' or
                'plot' method for visualization.

        """
        # Instantiate the analysis class with the optic and parameters
        analysis = self.config.analysis_class(
            self.optic, **self.config.analysis_params
        )

        # Generate the plot using either 'view' or 'plot' method
        if hasattr(analysis, "view"):
            fig, ax = analysis.view()
        elif hasattr(analysis, "plot"):
            fig, ax = analysis.plot()
        else:
            raise AttributeError(
                f"Analysis class {self.config.analysis_class.__name__} does "
                "not have a 'view' or 'plot' method."
            )

        # Configure the plot for clean rasterization
        axes = ax if isinstance(ax, list) else [ax]
        for axis in axes:
            axis.axis("off")
            axis.patch.set_alpha(1.0)  # Keep axes background opaque
            axis.set_facecolor("white") # Set a fixed background for visibility

        fig.tight_layout(pad=0)
        fig.patch.set_alpha(0.0)  # Make figure background transparent

        # Rasterize the figure to a NumPy array
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        image_array = np.asarray(rgba_buffer)
        # Keep only the RGB channels, discard alpha
        rgb_array = image_array[:, :, :3]
        plt.close(fig)  # Prevent the plot from being displayed interactively

        return rgb_array

    def _create_plot_actor(self):
        """Creates the textured vtkActor for the analysis plot."""
        # Get the plot as a high-resolution image array
        image_array = self._get_analysis_plot_as_array()
        height, width, _ = image_array.shape

        # Convert the NumPy array to VTK image data
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, 1)
        vtk_array = numpy_to_vtk(
            num_array=image_array.flatten(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_array.SetNumberOfComponents(3)
        vtk_image.GetPointData().SetScalars(vtk_array)

        # Create a VTK texture from the image data
        texture = vtk.vtkTexture()
        texture.SetInputData(vtk_image)
        texture.InterpolateOn()

        # Get the target surface's properties for positioning and scaling
        surface = self.optic.surface_group.surfaces[self.config.surface_index]
        radius = surface.semi_aperture or 10.0  # Use semi-aperture or a fallback

        # Create a plane source to serve as the canvas for the texture
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-radius, -radius, 0)
        plane.SetPoint1(radius, -radius, 0)
        plane.SetPoint2(-radius, radius, 0)

        # Get the effective transform of the surface's coordinate system
        translation, rotation = surface.geometry.cs.get_effective_transform()
        translation_np = be.to_numpy(translation)
        rotation_np = be.to_numpy(rotation)

        # Construct the 4x4 transformation matrix
        matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                matrix.SetElement(i, j, rotation_np[i, j])
            matrix.SetElement(i, 3, translation_np[i])
        matrix.SetElement(3, 3, 1)

        # Position and orient the plane using the constructed matrix
        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputConnection(plane.GetOutputPort())
        transform_filter.Update()

        # Create a mapper and set the actor's properties
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        self.actor.SetMapper(mapper)
        self.actor.SetTexture(texture)

    def plot(self, renderer: vtk.vtkRenderer):
        """Adds the configured plot actor to the given VTK renderer.

        Args:
            renderer: The vtkRenderer instance to which the actor will be
                added.

        """
        renderer.AddActor(self.actor)
