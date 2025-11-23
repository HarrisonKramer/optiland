import pytest
from optiland_gui.undo_redo_manager import UndoRedoManager

@pytest.fixture
def undo_manager(app):
    """Returns a new instance of UndoRedoManager for each test."""
    return UndoRedoManager()

def test_initial_state(undo_manager):
    """Test that the undo and redo stacks are initially empty."""
    assert not undo_manager.can_undo()
    assert not undo_manager.can_redo()

def test_add_state(undo_manager):
    """Test that adding a state enables undo and disables redo."""
    undo_manager.add_state("state1")
    assert undo_manager.can_undo()
    assert not undo_manager.can_redo()

def test_add_state_clears_redo_stack(undo_manager):
    """Test that adding a new state clears the redo stack."""
    undo_manager.add_state("state1")
    undo_manager.undo("state2")
    assert undo_manager.can_redo()

    undo_manager.add_state("state3")
    assert not undo_manager.can_redo()

def test_undo_empty_stack(undo_manager):
    """Test that undoing on an empty stack does nothing."""
    assert undo_manager.undo("current_state") is None
    assert not undo_manager.can_undo()
    assert not undo_manager.can_redo()

def test_redo_empty_stack(undo_manager):
    """Test that redoing on an empty stack does nothing."""
    assert undo_manager.redo("current_state") is None
    assert not undo_manager.can_undo()
    assert not undo_manager.can_redo()

def test_undo_redo_cycle(undo_manager):
    """Test a full undo/redo cycle."""
    undo_manager.add_state("state1")
    restored_state = undo_manager.undo("state2")
    assert restored_state == "state1"
    assert not undo_manager.can_undo()
    assert undo_manager.can_redo()

    restored_state = undo_manager.redo("state1")
    assert restored_state == "state2"
    assert undo_manager.can_undo()
    assert not undo_manager.can_redo()

def test_clear_stacks(undo_manager):
    """Test that clearing the stacks works correctly."""
    undo_manager.add_state("state1")
    undo_manager.undo("state2")
    assert undo_manager.can_undo() or undo_manager.can_redo()

    undo_manager.clear_stacks()
    assert not undo_manager.can_undo()
    assert not undo_manager.can_redo()

def test_undo_redo_signals(undo_manager, qtbot):
    """Test that the undo/redo signals are emitted correctly."""

    # Test add_state signals
    with qtbot.waitSignal(undo_manager.undoStackAvailabilityChanged) as undo_blocker:
        with qtbot.waitSignal(undo_manager.redoStackAvailabilityChanged) as redo_blocker:
            undo_manager.add_state("state1")
    assert undo_blocker.args == [True]
    assert redo_blocker.args == [False]

    # Test undo signals
    with qtbot.waitSignal(undo_manager.undoStackAvailabilityChanged) as undo_blocker:
        with qtbot.waitSignal(undo_manager.redoStackAvailabilityChanged) as redo_blocker:
            undo_manager.undo("state2")
    assert undo_blocker.args == [False]
    assert redo_blocker.args == [True]

    # Test redo signals
    with qtbot.waitSignal(undo_manager.undoStackAvailabilityChanged) as undo_blocker:
        with qtbot.waitSignal(undo_manager.redoStackAvailabilityChanged) as redo_blocker:
            undo_manager.redo("state1")
    assert undo_blocker.args == [True]
    assert redo_blocker.args == [False]

    # Test clear_stacks signals
    with qtbot.waitSignal(undo_manager.undoStackAvailabilityChanged) as undo_blocker:
        with qtbot.waitSignal(undo_manager.redoStackAvailabilityChanged) as redo_blocker:
            undo_manager.clear_stacks()
    assert undo_blocker.args == [False]
    assert redo_blocker.args == [False]

def test_undo_empty_stack_signals(undo_manager, qtbot):
    """Test that no signals are emitted when undoing on an empty stack."""
    with qtbot.assertNotEmitted(undo_manager.undoStackAvailabilityChanged):
        with qtbot.assertNotEmitted(undo_manager.redoStackAvailabilityChanged):
            undo_manager.undo("current_state")

def test_redo_empty_stack_signals(undo_manager, qtbot):
    """Test that no signals are emitted when redoing on an empty stack."""
    with qtbot.assertNotEmitted(undo_manager.undoStackAvailabilityChanged):
        with qtbot.assertNotEmitted(undo_manager.redoStackAvailabilityChanged):
            undo_manager.redo("current_state")
