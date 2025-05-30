# optiland_gui/lens_editor.py
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
                               QPushButton, QHBoxLayout, QAbstractItemView, QHeaderView)
from PySide6.QtCore import Qt
from .optiland_connector import OptilandConnector

class LensEditor(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Lens Editor")

        self.layout = QVBoxLayout(self)

        # Table for lens data
        self.tableWidget = QTableWidget()
        self.layout.addWidget(self.tableWidget)

        # Buttons for table manipulation
        self.buttonLayout = QHBoxLayout()
        self.btnAddSurface = QPushButton("Add Surface")
        self.btnRemoveSurface = QPushButton("Remove Surface")
        self.buttonLayout.addWidget(self.btnAddSurface)
        self.buttonLayout.addWidget(self.btnRemoveSurface)
        self.layout.addLayout(self.buttonLayout)

        self.setup_table()
        self.load_data()

        # Connect signals
        self.btnAddSurface.clicked.connect(self.add_surface)
        self.btnRemoveSurface.clicked.connect(self.remove_surface)
        self.tableWidget.itemChanged.connect(self.on_item_changed)

        self.connector.surfaceAdded.connect(self.handle_surface_added)
        self.connector.surfaceRemoved.connect(self.handle_surface_removed)
        self.connector.surfaceDataChanged.connect(self.handle_surface_data_changed)
        self.connector.surfaceCountChanged.connect(self.load_data) # Reload all on count change
        self.connector.opticChanged.connect(self.load_data) # Full reload if optic itself changes

    def setup_table(self):
        headers = self.connector.get_column_headers()
        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch) # Stretch columns

    def load_data(self):
        self.tableWidget.blockSignals(True) # Block signals during data loading
        self.tableWidget.setRowCount(0) # Clear existing rows
        num_surfaces = self.connector.get_surface_count()
        self.tableWidget.setRowCount(num_surfaces)

        for r in range(num_surfaces):
            for c in range(self.tableWidget.columnCount()):
                item_data = self.connector.get_surface_data(r, c)
                item = QTableWidgetItem(str(item_data) if item_data is not None else "")
                self.tableWidget.setItem(r, c, item)
        self.tableWidget.blockSignals(False)

    def on_item_changed(self, item: QTableWidgetItem):
        row = item.row()
        col = item.column()
        new_value = item.text()
        self.connector.set_surface_data(row, col, new_value)

    def add_surface(self):
        current_row = self.tableWidget.currentRow()
        # Add after current selection, or at the end if no selection or last row is selected
        insert_pos = current_row + 1 if current_row != -1 else self.tableWidget.rowCount()
        self.connector.add_surface(index=insert_pos) # Connector will emit signal to update table

    def remove_surface(self):
        current_row = self.tableWidget.currentRow()
        if current_row != -1:
            self.connector.remove_surface(current_row) # Connector will emit signal

    def handle_surface_added(self, index):
        # This might be redundant if opticChanged or surfaceCountChanged also calls load_data
        # self.load_data() # Simplest way to refresh
        pass # Relying on surfaceCountChanged or opticChanged for now

    def handle_surface_removed(self, index):
        # self.load_data() # Simplest way to refresh
        pass

    def handle_surface_data_changed(self, row, col):
        # self.tableWidget.blockSignals(True)
        # item_data = self.connector.get_surface_data(row, col)
        # self.tableWidget.item(row, col).setText(str(item_data) if item_data is not None else "")
        # self.tableWidget.blockSignals(False)
        pass # Item change already handled by on_item_changed, this signal is for other UI parts