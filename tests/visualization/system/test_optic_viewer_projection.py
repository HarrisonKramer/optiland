import pytest
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from optiland.optic import Optic
from optiland.samples import CookeTriplet
from optiland.visualization.system.optic_viewer import OpticViewer


@pytest.fixture
def optic() -> Optic:
    """A fixture that returns a CookeTriplet optic."""
    return CookeTriplet()


def test_optic_viewer_view_default_projection(optic: Optic):
    """Test that the default projection is YZ."""
    viewer = OpticViewer(optic)
    fig, ax, _ = viewer.view()
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Z [mm]"
    assert ax.get_ylabel() == "Y [mm]"
    plt.close(fig)


@pytest.mark.parametrize(
    "projection, expected_xlabel, expected_ylabel",
    [
        ("XY", "X [mm]", "Y [mm]"),
        ("XZ", "X [mm]", "Z [mm]"),
        ("YZ", "Z [mm]", "Y [mm]"),
    ],
)
def test_optic_viewer_view_valid_projections(
    optic: Optic, projection: str, expected_xlabel: str, expected_ylabel: str
):
    """Test that valid projections set the correct axis labels."""
    viewer = OpticViewer(optic)
    fig, ax, _ = viewer.view(projection=projection)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == expected_xlabel
    assert ax.get_ylabel() == expected_ylabel
    plt.close(fig)


def test_optic_viewer_view_invalid_projection(optic: Optic):
    """Test that an invalid projection raises a ValueError."""
    viewer = OpticViewer(optic)
    with pytest.raises(ValueError, match="Invalid projection type. Must be 'XY', 'XZ', or 'YZ'."):
        viewer.view(projection="invalid")
