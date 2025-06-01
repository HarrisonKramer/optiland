# optiland_gui/undo_redo_manager.py
from PySide6.QtCore import QObject, Signal


class UndoRedoManager(QObject):
    undoStackAvailabilityChanged = Signal(bool)
    redoStackAvailabilityChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._undo_stack = []
        self._redo_stack = []

    def add_state(self, state_data):
        """
        Adds a new state to the undo stack.
        This should be called before the action is performed.
        The 'state_data' is the state *before* the change.
        """
        self._undo_stack.append(state_data)
        self._redo_stack.clear()  # Clear redo stack whenever a new action is performed
        self.undoStackAvailabilityChanged.emit(self.can_undo())
        self.redoStackAvailabilityChanged.emit(self.can_redo())
        print(
            f"UndoRedoManager: State added. Undo stack size: {len(self._undo_stack)}, "
            f"Redo stack size: {len(self._redo_stack)}"
        )

    def undo(self, current_state_for_redo):
        """
        Performs an undo operation.
        Returns the state to restore.
        'current_state_for_redo' is the state *after* the original action was
        performed (i.e., the state *before* undoing).
        """
        if not self.can_undo():
            return None

        restored_state = self._undo_stack.pop()
        self._redo_stack.append(current_state_for_redo)

        self.undoStackAvailabilityChanged.emit(self.can_undo())
        self.redoStackAvailabilityChanged.emit(self.can_redo())
        print(
            f"UndoRedoManager: Undo. Undo stack size: {len(self._undo_stack)}, "
            f"Redo stack size: {len(self._redo_stack)}"
        )
        return restored_state

    def redo(self, current_state_for_undo):
        """
        Performs a redo operation.
        Returns the state to restore.
        'current_state_for_undo' is the state *after* the undo operation was
        performed (i.e., the state *before* redoing).
        """
        if not self.can_redo():
            return None

        restored_state = self._redo_stack.pop()
        # The 'current_state_for_undo' is the state that resulted from the 'undo'
        # operation. This state (which is about to be replaced by 'restored_state') is
        # what should go onto the undo stack.
        self._undo_stack.append(current_state_for_undo)

        self.undoStackAvailabilityChanged.emit(self.can_undo())
        self.redoStackAvailabilityChanged.emit(self.can_redo())
        print(
            f"UndoRedoManager: Redo. Undo stack size: {len(self._undo_stack)}, "
            f"Redo stack size: {len(self._redo_stack)}"
        )
        return restored_state

    def can_undo(self):
        return bool(self._undo_stack)

    def can_redo(self):
        return bool(self._redo_stack)

    def clear_stacks(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.undoStackAvailabilityChanged.emit(self.can_undo())
        self.redoStackAvailabilityChanged.emit(self.can_redo())
        print("UndoRedoManager: Stacks cleared.")
