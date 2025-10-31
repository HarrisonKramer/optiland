"""
Provides a manager for handling undo and redo functionality.

This module defines the `UndoRedoManager`, a class that maintains stacks of
object states to enable undoing and redoing actions within the application.

@authors: Originally wrote by Kramer Harrison, 2025
          Modified by Manuel Mendes,   2025
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class UndoRedoManager(QObject):
    """
    Manages undo and redo stacks for application state changes.

    This class provides a simple stack-based mechanism to manage states.
    It is designed to store snapshots of an object (e.g., an optical system)
    before a change occurs.

    Signals:
        undoStackAvailabilityChanged (bool): Emitted when the undo stack's
                                             availability changes.
        redoStackAvailabilityChanged (bool): Emitted when the redo stack's
                                             availability changes.

    Attributes:
        _undo_stack (list): A list of states that can be restored via undo.
        _redo_stack (list): A list of states that can be restored via redo.
    """

    undoStackAvailabilityChanged = Signal(bool)
    redoStackAvailabilityChanged = Signal(bool)

    def __init__(self, parent=None):
        """
        Initializes the UndoRedoManager.

        Args:
            parent (QObject, optional): The parent object. Defaults to None.
        """
        super().__init__(parent)
        self._undo_stack = []
        self._redo_stack = []

    def add_state(self, state_data):
        """
        Adds a new state to the undo stack.

        This should be called with the state of the object *before* a change
        is made. Calling this method clears the redo stack.

        Args:
            state_data: The data representing the state to be saved. This can be
                        any object that can be deep-copied.
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
        Performs an undo operation and returns the state to restore.

        The current state of the object is pushed onto the redo stack before
        the previous state is returned.

        Args:
            current_state_for_redo: The current state of the object *before*
                                    the undo operation is applied. This state
                                    will be added to the redo stack.

        Returns:
            The state to be restored, or None if the undo stack is empty.
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
        Performs a redo operation and returns the state to restore.

        The state of the object just before the redo is pushed onto the undo
        stack.

        Args:
            current_state_for_undo: The current state of the object *before*
                                    the redo operation is applied. This state
                                    will be added to the undo stack.

        Returns:
            The state to be restored, or None if the redo stack is empty.
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
        """
        Checks if there are any actions to undo.

        Returns:
            bool: True if the undo stack is not empty, False otherwise.
        """
        return bool(self._undo_stack)

    def can_redo(self):
        """
        Checks if there are any actions to redo.

        Returns:
            bool: True if the redo stack is not empty, False otherwise.
        """
        return bool(self._redo_stack)

    def clear_stacks(self):
        """Clears both the undo and redo stacks."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.undoStackAvailabilityChanged.emit(self.can_undo())
        self.redoStackAvailabilityChanged.emit(self.can_redo())
