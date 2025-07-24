"""Defines the LensEditor widget for displaying and editing optical system data.

This module contains the `LensEditor` class, a QWidget that provides a spreadsheet-like
interface (QTableWidget) for modifying the properties of an optical system's surfaces,
such as radius, thickness, and material.

Author: Manuel Fragata Mendes, 2025
"""

from PySide6.QtCore import QEvent, Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .optiland_connector import OptilandConnector


class LensEditor(QWidget):
    """A widget for editing the properties of an optical system's surfaces.

    The LensEditor displays optical system data in a table, allowing users to
    view and modify surface parameters. It interacts with an `OptilandConnector`
    to synchronize data with the underlying optical model. It also provides
    controls for adding and removing surfaces.

    Attributes:
        connector (OptilandConnector): The connector to the Optiland backend.
        tableWidget (QTableWidget): The table displaying the lens data.
    """

    def __init__(self, connector: OptilandConnector, parent=None):
        """Initializes the LensEditor widget.

        Args:
            connector: The `OptilandConnector` instance for backend communication.
            parent: The parent widget, if any.
        """
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Lens Editor")

        self.layout = QVBoxLayout(self)

        self.tableWidget = QTableWidget()
        self.tableWidget.installEventFilter(self)
        self.layout.addWidget(self.tableWidget)

        self.buttonLayout = QHBoxLayout()
        self.btnAddSurface = QPushButton("Add Surface")
        self.btnRemoveSurface = QPushButton("Remove Surface")
        self.buttonLayout.addWidget(self.btnAddSurface)
        self.buttonLayout.addWidget(self.btnRemoveSurface)
        self.layout.addLayout(self.buttonLayout)

        self.setup_table()
        self.load_data()

        self.btnAddSurface.clicked.connect(self.add_surface_handler)
        self.btnRemoveSurface.clicked.connect(self.remove_surface_handler)
        self.tableWidget.itemChanged.connect(self.on_item_changed_handler)

        self.connector.opticLoaded.connect(self.full_refresh_from_optic)
        self.connector.surfaceDataChanged.connect(self.update_cell_from_connector)
        self.connector.surfaceCountChanged.connect(self.full_refresh_from_optic)
        self.connector.opticChanged.connect(self.full_refresh_from_optic)

    def setup_table(self):
        """Sets up the table's headers and initial configuration.

        This method configures the table's column headers based on data from the
        connector and sets initial properties like selection behavior and resize modes.
        """
        self.tableWidget.blockSignals(True)
        headers = self.connector.get_column_headers()
        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)
        self.tableWidget.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.tableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.tableWidget.blockSignals(False)

    def eventFilter(self, source, event):
        """Intercepts key presses on the table to handle shortcuts.

        This event filter listens for `Insert` and `Delete` key presses on the
        table widget to trigger the addition or removal of surfaces, providing
        keyboard shortcuts for these common actions.

        Args:
            source: The object that generated the event.
            event: The event that occurred.

        Returns:
            bool: True if the event was handled, otherwise calls the base
                    class implementation.
        """
        if source is self.tableWidget and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Insert:
                self.add_surface_handler()
                return True
            elif event.key() == Qt.Key_Delete:
                self.remove_surface_handler()
                return True

        return super().eventFilter(source, event)

    @Slot()
    def full_refresh_from_optic(self):
        """Fully refreshes the editor by reloading all data from the optic.

        This slot is connected to signals indicating a major change in the optical
        system (e.g., a new file loaded, surface count changed). It re-initializes
        the table structure and reloads all data.
        """
        self.setup_table()
        self.load_data()

    @Slot()
    def load_data(self):
        """Populates the table with the current surface data from the connector.

        This method clears the table and fills it with the latest data from the
        optical system via the `OptilandConnector`. It also sets item flags to
        make certain cells (like object/image surface properties) non-editable.
        """
        self.tableWidget.blockSignals(True)
        self.tableWidget.setRowCount(0)
        num_surfaces = self.connector.get_surface_count()
        self.tableWidget.setRowCount(num_surfaces)

        headers = self.connector.get_column_headers()
        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)

        for r in range(num_surfaces):
            for c_idx, header in enumerate(headers):
                item_data = self.connector.get_surface_data(r, c_idx)
                item = QTableWidgetItem(str(item_data) if item_data is not None else "")
                is_obj_or_img = r == 0 or r == num_surfaces - 1
                if is_obj_or_img and header in [
                    "Radius",
                    "Thickness",
                    "Material",
                    "Conic",
                    "Semi-Diameter",
                ]:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if r == num_surfaces - 1 and header == "Thickness":
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                self.tableWidget.setItem(r, c_idx, item)
        self.tableWidget.blockSignals(False)

    @Slot(QTableWidgetItem)
    def on_item_changed_handler(self, item: QTableWidgetItem):
        """Handles changes made to a table item by the user.

        This slot is triggered when the user edits a cell in the table. It sends
        the updated value to the `OptilandConnector` to modify the underlying
        optical model.

        Args:
            item: The `QTableWidgetItem` that was changed.
        """
        if not self.tableWidget.signalsBlocked():
            row = item.row()
            col = item.column()
            new_value_str = item.text()
            print(
                f"LensEditor: Item changed by user at "
                f"({row},{col}) to '{new_value_str}'"
            )
            self.connector.set_surface_data(row, col, new_value_str)

    @Slot(int, int, object)
    def update_cell_from_connector(self, row, col, new_value_display_text):
        """Updates a single cell in the table with a new value from the backend.

        This slot is connected to a signal from the `OptilandConnector` and is
        called when a property changes in the backend. This ensures the UI stays
        in sync with the model state.

        Args:
            row (int): The row index of the cell to update.
            col (int): The column index of the cell to update.
            new_value_display_text (object): The new value to display in the cell.
        """
        self.tableWidget.blockSignals(True)
        if (
            0 <= row < self.tableWidget.rowCount()
            and 0 <= col < self.tableWidget.columnCount()
        ):
            item = self.tableWidget.item(row, col)
            if item:
                item.setText(str(new_value_display_text))
            else:
                new_item = QTableWidgetItem(str(new_value_display_text))
                self.tableWidget.setItem(row, col, new_item)
        self.tableWidget.blockSignals(False)

    @Slot()
    def add_surface_handler(self):
        """Adds a new surface to the optical system.

        This slot is connected to the 'Add Surface' button. It determines the
        correct insertion index based on the user's selection and requests the
        `OptilandConnector` to add a new surface at that position.
        """
        current_row = self.tableWidget.currentRow()
        insert_pos_lde = (
            current_row + 1 if current_row != -1 else self.tableWidget.rowCount()
        )
        self.connector.add_surface(index=insert_pos_lde)

    @Slot()
    def remove_surface_handler(self):
        """Removes the selected surface from the optical system.

        This slot is connected to the 'Remove Surface' button. It removes the
        currently selected row (surface) from the optical system via the
        `OptilandConnector`.
        """
        current_row = self.tableWidget.currentRow()
        if current_row != -1:
            self.connector.remove_surface(current_row)
