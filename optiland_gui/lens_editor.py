# optiland_gui/lens_editor.py
from PySide6.QtCore import Qt, Slot
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
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Lens Editor")

        self.layout = QVBoxLayout(self)

        self.tableWidget = QTableWidget()
        self.layout.addWidget(self.tableWidget)

        self.buttonLayout = QHBoxLayout()
        self.btnAddSurface = QPushButton("Add Surface")
        self.btnRemoveSurface = QPushButton("Remove Surface")
        self.buttonLayout.addWidget(self.btnAddSurface)
        self.buttonLayout.addWidget(self.btnRemoveSurface)
        self.layout.addLayout(self.buttonLayout)

        self.setup_table()  # Setup is now dynamic based on connector
        self.load_data()  # Initial data load

        # Connect signals
        self.btnAddSurface.clicked.connect(self.add_surface_handler)
        self.btnRemoveSurface.clicked.connect(self.remove_surface_handler)
        self.tableWidget.itemChanged.connect(self.on_item_changed_handler)

        # Connector signals for table updates
        self.connector.opticLoaded.connect(self.full_refresh_from_optic)
        self.connector.surfaceDataChanged.connect(self.update_cell_from_connector)
        # surfaceAdded and surfaceRemoved will trigger surfaceCountChanged
        self.connector.surfaceCountChanged.connect(self.full_refresh_from_optic)

    def setup_table(self):
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
        # self.tableWidget.verticalHeader().
        # setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tableWidget.blockSignals(False)

    @Slot()
    def full_refresh_from_optic(self):
        print("LensEditor: Full refresh from optic signal received.")
        self.setup_table()  # Re-setup in case columns changed (though unlikely here)
        self.load_data()

    @Slot()
    def load_data(self):
        self.tableWidget.blockSignals(True)
        self.tableWidget.setRowCount(0)
        num_surfaces = self.connector.get_surface_count()
        self.tableWidget.setRowCount(num_surfaces)

        headers = (
            self.connector.get_column_headers()
        )  # Ensure we use current headers for column count
        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)

        for r in range(num_surfaces):
            for c_idx in range(len(headers)):
                item_data = self.connector.get_surface_data(r, c_idx)
                item = QTableWidgetItem(str(item_data) if item_data is not None else "")
                # Make Object/Image surface types non-editable for certain fields
                is_obj_or_img = r == 0 or r == num_surfaces - 1
                if is_obj_or_img and headers[c_idx] in [
                    "Radius",
                    "Thickness",
                    "Material",
                    "Conic",
                    "Semi-Diameter",
                ]:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if (
                    r == num_surfaces - 1 and headers[c_idx] == "Thickness"
                ):  # Last surface thickness
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                self.tableWidget.setItem(r, c_idx, item)
        self.tableWidget.blockSignals(False)

    @Slot(QTableWidgetItem)
    def on_item_changed_handler(self, item: QTableWidgetItem):
        if not self.tableWidget.signalsBlocked():  # Check if change is user-initiated
            row = item.row()
            col = item.column()
            new_value_str = item.text()
            print(
                f"LensEditor: Item changed by user at "
                f"({row},{col}) to '{new_value_str}'"
            )
            self.connector.set_surface_data(row, col, new_value_str)
            # Data will be re-fetched and reformatted by update_cell_from_connector
            # or full_refresh_from_optic if opticChanged is broad.

    @Slot(int, int, object)
    def update_cell_from_connector(self, row, col, new_value_display_text):
        self.tableWidget.blockSignals(True)
        # Ensure row and col are valid before trying to access item
        if (
            0 <= row < self.tableWidget.rowCount()
            and 0 <= col < self.tableWidget.columnCount()
        ):
            item = self.tableWidget.item(row, col)
            if item:
                item.setText(str(new_value_display_text))
            else:  # Should not happen if table is synced
                new_item = QTableWidgetItem(str(new_value_display_text))
                self.tableWidget.setItem(row, col, new_item)
        self.tableWidget.blockSignals(False)

    @Slot()
    def add_surface_handler(self):
        current_row = self.tableWidget.currentRow()
        insert_pos_lde = (
            current_row + 1 if current_row != -1 else self.tableWidget.rowCount()
        )
        # Let OptilandConnector handle the logic of where to insert in the Optic object
        self.connector.add_surface(index=insert_pos_lde)

    @Slot()
    def remove_surface_handler(self):
        current_row = self.tableWidget.currentRow()
        if current_row != -1:
            # Add confirmation dialog here if desired
            self.connector.remove_surface(current_row)
