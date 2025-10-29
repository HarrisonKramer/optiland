"""Unit tests for the visualization interaction module.

"""

import pytest
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, PickEvent

from optiland.samples import CookeTriplet
from optiland.visualization.system.surface import Surface2D
from optiland.visualization.system.optic_viewer import OpticViewer


@pytest.fixture
def optic_viewer():
    """Returns an OpticViewer instance for a Cooke Triplet."""
    optic = CookeTriplet()
    return OpticViewer(optic)


def test_interaction_manager_initialization(optic_viewer):
    """Tests that the InteractionManager is initialized with the correct figure and axes."""
    fig, ax, im = optic_viewer.view()
    assert im.fig == fig
    assert im.ax == ax
    plt.close(fig)


def test_hover_highlight_and_tooltip(optic_viewer):
    """Tests that hovering over a surface highlights it and shows a tooltip."""
    fig, ax, im = optic_viewer.view()

    # Find a surface artist to hover over
    surface_artist = None
    for artist, obj in im.artist_registry.items():
        if isinstance(obj, Surface2D):
            surface_artist = artist
            break
    assert surface_artist is not None

    # Get the center of the artist
    x_data = surface_artist.get_xdata()
    y_data = surface_artist.get_ydata()
    x, y = x_data[len(x_data)//2], y_data[len(y_data)//2]
    x_pix, y_pix = ax.transData.transform((x, y))

    # Simulate a hover event
    event = MouseEvent('motion_notify_event', fig.canvas, x_pix, y_pix)
    event.inaxes = ax
    im.on_hover(event)

    # Check that the artist is highlighted
    assert surface_artist.get_linewidth() > 1

    # Check that the tooltip is visible and has the correct text
    assert im.tooltip.get_visible()
    assert "Surface" in im.tooltip.get_text()

    plt.close(fig)


def test_click_to_inspect(optic_viewer):
    """Tests that clicking on a surface shows the information panel."""
    fig, ax, im = optic_viewer.view()

    # Find a surface artist to click on
    surface_artist = None
    for artist, obj in im.artist_registry.items():
        if isinstance(obj, Surface2D):
            surface_artist = artist
            break
    assert surface_artist is not None

    # Get the center of the artist
    x_data = surface_artist.get_xdata()
    y_data = surface_artist.get_ydata()
    x, y = x_data[len(x_data)//2], y_data[len(y_data)//2]
    x_pix, y_pix = ax.transData.transform((x, y))

    # Simulate a click event
    event = MouseEvent('button_press_event', fig.canvas, x_pix, y_pix)
    event.inaxes = ax
    im.on_click(event)

    # Check that the info panel is visible
    assert im.info_panel is not None
    assert im.info_panel.get_visible()

    plt.close(fig)


