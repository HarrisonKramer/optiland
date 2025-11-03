"""Unit tests for the 3D surface plot framework."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from optiland.analysis import SpotDiagram
from optiland.samples import CookeTriplet
from optiland.visualization.system.surface_plot import SurfacePlot


@pytest.mark.parametrize("be_in", ["numpy"])
def test_draw3d_with_surface_plot(be_in, set_test_backend):
    """
    Verify that calling Optic.draw3D with a SurfacePlot configuration
    results in a call to the renderer's AddActor method.
    """
    # 1. Arrange
    optic = CookeTriplet()
    surface_plot_config = SurfacePlot(
        surface_index=-1,  # Image plane
        analysis_class=SpotDiagram,
        analysis_params={"num_rings": 5},
    )

    # Mock the core VTK classes to prevent any actual rendering
    with patch('vtk.vtkRenderWindowInteractor') as mock_interactor_cls, \
         patch('vtk.vtkRenderer') as mock_renderer_cls, \
         patch('vtk.vtkRenderWindow'):  # Also mock RenderWindow

        mock_renderer_instance = mock_renderer_cls.return_value
        mock_interactor_instance = mock_interactor_cls.return_value

        # 2. Act
        # Calling draw3D should trigger the new surface plot logic,
        # using our mock VTK instances instead of real ones.
        optic.draw3D(surface_plots=[surface_plot_config])

        # 3. Assert
        # Check that the interactor was started, confirming the render
        # pipeline was executed.
        mock_interactor_instance.Start.assert_called_once()

        # Check that actors were added to the renderer.
        assert mock_renderer_instance.AddActor.call_count > 0

        # Perform a more specific check to ensure the plot actor was added.
        # We identify the plot actor by checking if it has a texture applied.
        found_plot_actor = False
        for call in mock_renderer_instance.AddActor.call_args_list:
            actor = call.args[0]
            # The plot actor is the only one that should have a texture
            if hasattr(actor, "GetTexture") and actor.GetTexture() is not None:
                found_plot_actor = True
                break

        assert found_plot_actor, "The expected plot actor with a texture was not added to the renderer."
