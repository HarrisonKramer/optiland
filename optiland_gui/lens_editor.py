"""Defines the LensEditor widget for displaying and editing optical system data.

This module contains the `LensEditor` class, a QWidget that provides a spreadsheet-like
interface (QTableWidget) for modifying the properties of an optical system's surfaces,
such as radius, thickness, and material.

Author: Manuel Fragata Mendes, 2025
"""

from PySide6.QtCore import QEvent, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .optiland_connector import OptilandConnector


class SurfacePropertiesWidget(QWidget):
    """A widget to display and edit specific parameters of a surface geometry."""

    def __init__(self, row, connector, parent=None):
        super().__init__(parent)
        self.row = row
        self.connector = connector
        self.setObjectName("SurfacePropertiesWidget")
        self.setMinimumWidth(750)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 8, 15, 8)
        self.form_layout = QFormLayout()
        self.form_layout.setHorizontalSpacing(15)
        self.form_layout.setVerticalSpacing(5)
        main_layout.addLayout(self.form_layout)

        self.input_widgets = {}
        params = self.connector.get_surface_geometry_params(self.row)

        if not params:
            self.form_layout.addRow(
                QLabel("No additional properties for this surface type.")
            )

        for name, value in params.items():
            label_text = name + ":"
            line_edit = QLineEdit()
            line_edit.setMaximumWidth(60)  # Shorter text boxes

            if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
                list_val = value.tolist() if hasattr(value, "tolist") else value
                line_edit.setText(str(list_val))
                line_edit.setPlaceholderText("e.g., [0.1, -0.2]")
            else:
                line_edit.setText(f"{value:.6f}")

            line_edit.editingFinished.connect(self.apply_changes)  # Auto-apply
            self.form_layout.addRow(label_text, line_edit)
            self.input_widgets[name] = line_edit

    @Slot()
    def apply_changes(self):
        """Collects data from input fields and sends it to the connector."""
        params_to_set = {}
        for name, widget in self.input_widgets.items():
            params_to_set[name] = widget.text()
        self.connector.set_surface_geometry_params(self.row, params_to_set)


class SurfaceTypeWidget(QWidget):
    """A custom widget for the 'Type' column, allowing text edit and dropdown."""

    surfaceTypeChanged = Signal(str)

    def __init__(self, row, current_type_info, connector, parent=None):
        super().__init__(parent)
        self.row = row
        self.connector = connector
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(2, 0, 2, 0)
        self.layout.setSpacing(4)
        self.type_button = QToolButton()
        self.type_button.setObjectName("SurfaceTypeButton")
        self.type_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.type_button.setFixedSize(18, 18)
        self.type_button.setAutoRaise(True)
        self.type_button.setArrowType(Qt.DownArrow)
        self.type_edit = QLineEdit(current_type_info["display_text"])
        self.type_edit.setObjectName("SurfaceTypeLineEdit")
        self.type_edit.editingFinished.connect(self.text_changed)
        self.layout.addWidget(self.type_edit, 1)
        self.layout.addWidget(self.type_button)
        self.surface_menu = QMenu(self)
        self.surface_menu.setObjectName("SurfaceTypeMenu")
        for surf_type in self.connector.get_available_surface_types():
            action = self.surface_menu.addAction(surf_type.title())
            action.triggered.connect(
                lambda checked=False, t=surf_type: self.type_selected(t)
            )
        self.type_button.setMenu(self.surface_menu)
        is_editable = current_type_info["is_changeable"]
        self.type_button.setEnabled(is_editable)
        self.type_button.setVisible(is_editable)
        self.type_edit.setReadOnly(not is_editable)

    def type_selected(self, new_type):
        self.type_edit.setText(new_type.title())
        self.surfaceTypeChanged.emit(new_type)

    def text_changed(self):
        new_type = self.type_edit.text()
        if new_type.lower().strip() in self.connector.get_available_surface_types():
            self.surfaceTypeChanged.emit(new_type)
        else:
            type_info = self.connector.get_surface_type_info(self.row)
            self.type_edit.setText(type_info["display_text"])


class LensEditor(QWidget):
    """A widget for editing the properties of an optical system's surfaces."""

    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Lens Editor")
        self.open_prop_source_row = -1
        self.layout = QVBoxLayout(self)
        self.tableWidget = QTableWidget()
        self.tableWidget.installEventFilter(self)
        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.layout.addWidget(self.tableWidget)
        self.buttonLayout = QHBoxLayout()
        self.btnAddSurface = QPushButton("Add Surface")
        self.btnRemoveSurface = QPushButton("Remove Surface")
        self.buttonLayout.addWidget(self.btnAddSurface)
        self.buttonLayout.addWidget(self.btnRemoveSurface)
        self.layout.addLayout(self.buttonLayout)
        self.setup_table()
        self.load_data()
        self.connect_signals()

    def connect_signals(self):
        self.btnAddSurface.clicked.connect(self.add_surface_handler)
        self.btnRemoveSurface.clicked.connect(self.remove_surface_handler)
        self.tableWidget.itemChanged.connect(self.on_item_changed_handler)
        self.tableWidget.customContextMenuRequested.connect(self.show_context_menu)
        self.tableWidget.itemSelectionChanged.connect(self.update_headers_on_selection)
        self.connector.opticLoaded.connect(self.full_refresh_from_optic)
        self.connector.opticChanged.connect(self.full_refresh_from_optic)

    def setup_table(self):
        self.tableWidget.blockSignals(True)
        self.tableWidget.setColumnCount(len(self.connector.get_column_headers()))
        self.tableWidget.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.tableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.tableWidget.blockSignals(False)

    def eventFilter(self, source, event):
        if source is self.tableWidget and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Insert:
                self.add_surface_handler()
                return True
            if event.key() == Qt.Key_Delete:
                self.remove_surface_handler()
                return True
        return super().eventFilter(source, event)

    def map_ui_row_to_surface_index(self, ui_row):
        if self.open_prop_source_row != -1 and ui_row > self.open_prop_source_row:
            return ui_row - 1
        return ui_row

    def map_surface_index_to_ui_row(self, surface_index):
        if (
            self.open_prop_source_row != -1
            and surface_index > self.open_prop_source_row
        ):
            return surface_index + 1
        return surface_index

    @Slot()
    def full_refresh_from_optic(self):
        self.load_data()
        self.update_headers_on_selection()

    @Slot()
    def load_data(self):
        self.tableWidget.blockSignals(True)
        self.tableWidget.setRowCount(0)
        num_surfaces = self.connector.get_surface_count()
        self.tableWidget.setRowCount(num_surfaces)

        for r in range(num_surfaces):
            self.tableWidget.setVerticalHeaderItem(r, QTableWidgetItem(str(r + 1)))
            for c_idx, header in enumerate(self.connector.get_column_headers()):
                if c_idx == self.connector.COL_TYPE:
                    type_info = self.connector.get_surface_type_info(r)
                    widget = SurfaceTypeWidget(r, type_info, self.connector)
                    widget.surfaceTypeChanged.connect(
                        lambda nt, row=r: self.connector.set_surface_type(row, nt)
                    )
                    self.tableWidget.setCellWidget(r, c_idx, widget)
                else:
                    item_data = self.connector.get_surface_data(r, c_idx)
                    item = QTableWidgetItem(
                        str(item_data) if item_data is not None else ""
                    )
                    is_obj_or_img = r == 0 or r == num_surfaces - 1
                    if (
                        is_obj_or_img
                        and header
                        in ["Radius", "Thickness", "Material", "Conic", "Semi-Diameter"]
                    ) or (r == num_surfaces - 1 and header == "Thickness"):
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.tableWidget.setItem(r, c_idx, item)

        if self.open_prop_source_row != -1 and self.open_prop_source_row < num_surfaces:
            self._insert_properties_widget(self.open_prop_source_row)

        self.tableWidget.blockSignals(False)

    def _insert_properties_widget(self, source_row):
        prop_row_index = source_row + 1
        self.tableWidget.insertRow(prop_row_index)
        self.tableWidget.setVerticalHeaderItem(prop_row_index, QTableWidgetItem(""))
        prop_widget = SurfacePropertiesWidget(source_row, self.connector)
        self.tableWidget.setCellWidget(prop_row_index, 0, prop_widget)
        self.tableWidget.setSpan(prop_row_index, 0, 1, self.tableWidget.columnCount())
        self.tableWidget.resizeRowToContents(prop_row_index)

    @Slot()
    def update_headers_on_selection(self):
        selected_items = self.tableWidget.selectedItems()
        row = (
            self.tableWidget.currentRow()
            if not selected_items
            else selected_items[0].row()
        )
        surface_index = self.map_ui_row_to_surface_index(row)
        headers = self.connector.get_column_headers(surface_index)
        self.tableWidget.setHorizontalHeaderLabels(headers)

    @Slot(QTableWidgetItem)
    def on_item_changed_handler(self, item: QTableWidgetItem):
        if not self.tableWidget.signalsBlocked():
            surface_index = self.map_ui_row_to_surface_index(item.row())
            self.connector.set_surface_data(surface_index, item.column(), item.text())

    @Slot()
    def add_surface_handler(self, surface_index_to_add_before=None):
        if surface_index_to_add_before is not None:
            self.connector.add_surface(index=surface_index_to_add_before)
        else:
            ui_row = self.tableWidget.currentRow()
            surface_index = self.map_ui_row_to_surface_index(ui_row)
            insert_pos = (
                surface_index + 1
                if ui_row != -1
                else self.connector.get_surface_count() - 1
            )
            self.connector.add_surface(index=insert_pos)

    @Slot()
    def remove_surface_handler(self, surface_index_to_remove=None):
        if surface_index_to_remove is None:
            ui_row = self.tableWidget.currentRow()
            if ui_row == -1:
                return
            surface_index_to_remove = self.map_ui_row_to_surface_index(ui_row)

        if self.open_prop_source_row == surface_index_to_remove:
            self.open_prop_source_row = -1  # Close properties if its owner is removed

        self.connector.remove_surface(surface_index_to_remove)

    @Slot()
    def toggle_properties_widget(self, source_row):
        self.open_prop_source_row = (
            -1 if self.open_prop_source_row == source_row else source_row
        )
        self.load_data()

    @Slot("QPoint")
    def show_context_menu(self, pos):
        ui_row = self.tableWidget.rowAt(pos.y())
        if ui_row < 0:
            return

        is_prop_widget_row = (
            self.open_prop_source_row != -1 and ui_row == self.open_prop_source_row + 1
        )

        surface_index = self.map_ui_row_to_surface_index(ui_row)

        menu = QMenu(self)
        menu.setObjectName("LDEContextMenu")

        if not is_prop_widget_row:
            add_above = menu.addAction("Add Surface Above")
            add_above.triggered.connect(lambda: self.add_surface_handler(surface_index))
            remove_action = menu.addAction("Remove Current Surface")
            remove_action.triggered.connect(
                lambda: self.remove_surface_handler(surface_index)
            )
            menu.addSeparator()
            props_action = menu.addAction("Surface Properties")
            props_action.triggered.connect(
                lambda: self.toggle_properties_widget(surface_index)
            )
            editor_action = menu.addAction("Surface Editor (WIP)")
            editor_action.setEnabled(False)

            is_obj_or_img = (surface_index == 0) or (
                surface_index == self.connector.get_surface_count() - 1
            )
            if is_obj_or_img:
                if surface_index == 0:
                    add_above.setEnabled(False)
                remove_action.setEnabled(False)
                props_action.setEnabled(False)

        menu.exec(self.tableWidget.viewport().mapToGlobal(pos))
