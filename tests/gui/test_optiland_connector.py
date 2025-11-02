import pytest
from unittest.mock import patch, MagicMock

from optiland.optic.optic import Optic
from optiland_gui.optiland_connector import OptilandConnector


@pytest.fixture
def connector(qtbot):
    """Fixture for a standalone OptilandConnector."""
    with patch("optiland_gui.optiland_connector.QMessageBox"):
        yield OptilandConnector()


def test_optiland_connector_init(connector):
    """Test the initialization of the OptilandConnector."""
    assert isinstance(connector, OptilandConnector)
    assert connector.get_optic() is not None
    assert connector._undo_redo_manager is not None
    assert connector.get_optic().name == "Default System"
    assert connector.get_surface_count() == 3
    assert not connector.is_modified()


def test_set_modified(connector, qtbot):
    """Test the set_modified method and modifiedStateChanged signal."""
    connector.set_modified(False)  # Ensure initial state

    with qtbot.wait_signal(connector.modifiedStateChanged, timeout=1000) as blocker:
        connector.set_modified(True)
    assert blocker.args == [True]
    assert connector.is_modified() is True

    with qtbot.wait_signal(connector.modifiedStateChanged, timeout=1000) as blocker:
        connector.set_modified(False)
    assert blocker.args == [False]
    assert connector.is_modified() is False

    with qtbot.assertNotEmitted(connector.modifiedStateChanged):
        connector.set_modified(False)


def test_new_system(connector, qtbot):
    """Test creating a new system."""
    connector._current_filepath = "/some/dummy/path.json"
    connector.set_modified(True)

    with qtbot.wait_signals([connector.opticLoaded, connector.opticChanged]):
        connector.new_system()

    assert connector.get_optic().name == "New Untitled System"
    assert connector.get_surface_count() == 3
    assert connector.get_current_filepath() is None
    assert not connector.is_modified()


def test_load_optic_from_object(connector, qtbot):
    """Test loading an optic from an object."""
    new_optic = Optic(name="My Test Optic")
    new_optic.add_surface(
        index=0, surface_type='standard', radius=float('inf'),
        thickness=float('inf'), comment='Object', material='Air'
    )
    new_optic.add_surface(
        index=1, surface_type='standard', radius=float('inf'),
        thickness=20.0, comment='Stop', is_stop=True, material='Air'
    )
    new_optic.add_surface(
        index=2, surface_type='standard', radius=float('inf'),
        thickness=0.0, comment='Image', material='Air'
    )
    new_optic.add_wavelength(0.55, is_primary=True)
    new_optic.set_field_type('angle')
    new_optic.add_field(y=0)
    new_optic.set_aperture('EPD', 10.0)

    with qtbot.wait_signals([connector.opticLoaded, connector.opticChanged]):
        connector.load_optic_from_object(new_optic)

    assert connector.get_optic().name == "My Test Optic"
    assert connector.is_modified()


@patch("optiland_gui.optiland_connector.QMessageBox")
def test_load_optic_from_object_error(mock_qmessagebox):
    """Test that loading a faulty optic object shows an error and resets."""
    connector = OptilandConnector()
    bad_optic = MagicMock(spec=Optic)
    bad_optic.to_dict.side_effect = Exception("Test Serialization Error")

    connector.load_optic_from_object(bad_optic)

    mock_qmessagebox.critical.assert_called_once()
    assert connector.get_optic().name == "New Untitled System"


def test_get_surface_count(connector):
    """Test getting the surface count."""
    assert connector.get_surface_count() == 3
    connector.add_surface()
    assert connector.get_surface_count() == 4


def test_get_column_headers(connector):
    """Test getting column headers."""
    headers = connector.get_column_headers()
    assert "Radius" in headers
    # Paraxial surface should have "Focal Length" instead of "Radius"
    connector.set_surface_type(1, "paraxial")
    headers = connector.get_column_headers(row=1)
    assert "Focal Length" in headers


def test_get_available_surface_types(connector):
    """Test getting available surface types."""
    types = connector.get_available_surface_types()
    assert "standard" in types
    assert "paraxial" in types


def test_get_and_set_surface_data(connector, qtbot):
    """Test getting and setting surface data."""
    # Test getting initial data
    assert connector.get_surface_data(1, connector.COL_COMMENT) == "Stop"

    # Test setting data
    with qtbot.wait_signal(connector.opticChanged):
        connector.set_surface_data(1, connector.COL_COMMENT, "New Comment")

    assert connector.get_surface_data(1, connector.COL_COMMENT) == "New Comment"


def test_add_and_remove_surface(connector, qtbot):
    """Test adding and removing a surface."""
    initial_count = connector.get_surface_count()

    with qtbot.wait_signal(connector.opticChanged):
        connector.add_surface(index=2)
    assert connector.get_surface_count() == initial_count + 1
    assert connector.get_surface_data(2, connector.COL_COMMENT) == "New Surface"

    with qtbot.wait_signal(connector.opticChanged):
        connector.remove_surface(2)
    assert connector.get_surface_count() == initial_count


def test_undo_redo(connector):
    """Test the undo and redo functionality."""
    original_comment = connector.get_surface_data(1, connector.COL_COMMENT)
    assert original_comment == "Stop"

    # Make a change
    connector.set_surface_data(1, connector.COL_COMMENT, "Changed Comment")
    assert connector.get_surface_data(1, connector.COL_COMMENT) == "Changed Comment"

    # Undo the change
    connector.undo()
    assert connector.get_surface_data(1, connector.COL_COMMENT) == original_comment

    # Redo the change
    connector.redo()
    assert connector.get_surface_data(1, connector.COL_COMMENT) == "Changed Comment"
