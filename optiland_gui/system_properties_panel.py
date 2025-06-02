# optiland_gui/system_properties_panel.py
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .optiland_connector import OptilandConnector


class SystemPropertiesPanel(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("System Properties")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Navigation Tree (like System Explorer)
        self.navTree = QTreeWidget()
        self.navTree.setHeaderHidden(True)
        self.navTree.setFixedWidth(150)  # Adjust as needed
        main_layout.addWidget(self.navTree)

        # StackedWidget to hold different property editors
        self.stackedWidget = QStackedWidget()
        main_layout.addWidget(self.stackedWidget)

        # --- Create and add property editor pages ---
        self.apertureEditor = ApertureEditor(self.connector)
        self.fieldsEditor = FieldsEditor(self.connector)
        self.wavelengthsEditor = WavelengthsEditor(self.connector)
        # Add more editors here (Environment, Polarization, etc. as placeholders)

        self.add_nav_item("Aperture", self.apertureEditor)
        self.add_nav_item("Fields", self.fieldsEditor)
        self.add_nav_item("Wavelengths", self.wavelengthsEditor)

        # Placeholder for other categories from screenshot
        for placeholder_name in [
            "Environment",
            "Polarization",
            "Advanced",
            "Ray Aiming",
            "Material Catalogs",
            "Title/Notes",
            "Files",
            "Units",
            "Cost Estimator",
        ]:
            placeholder_widget = QWidget()
            placeholder_layout = QVBoxLayout(placeholder_widget)
            placeholder_layout.addWidget(
                QLabel(f"{placeholder_name} Properties (Placeholder)")
            )
            placeholder_layout.addStretch()
            self.add_nav_item(placeholder_name, placeholder_widget)

        self.navTree.itemClicked.connect(self.on_nav_item_clicked)
        self.navTree.expandAll()
        if self.navTree.topLevelItemCount() > 0:  # Select first item by default
            self.navTree.setCurrentItem(self.navTree.topLevelItem(0))
            self.stackedWidget.setCurrentIndex(0)

        self.connector.opticLoaded.connect(self.load_properties)
        self.connector.opticChanged.connect(self.load_properties)  # General refresh

    def add_nav_item(self, name, widget):
        item = QTreeWidgetItem(self.navTree, [name])
        self.stackedWidget.addWidget(widget)
        item.setData(
            0, Qt.ItemDataRole.UserRole, self.stackedWidget.count() - 1
        )  # Store index

    @Slot(QTreeWidgetItem, int)
    def on_nav_item_clicked(self, item, column):
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if index is not None:
            self.stackedWidget.setCurrentIndex(index)

    @Slot()
    def load_properties(self):
        self.apertureEditor.load_data()
        self.fieldsEditor.load_data()
        self.wavelengthsEditor.load_data()
        # Call load_data for other editors when implemented


class ApertureEditor(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.is_loading = False

        layout = QFormLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.cmbApertureType = QComboBox()
        self.cmbApertureType.addItems(
            ["EPD", "imageFNO", "objectNA", "float_by_stop_size"]
        )
        layout.addRow("Aperture Type:", self.cmbApertureType)

        self.spnApertureValue = QDoubleSpinBox()
        self.spnApertureValue.setDecimals(4)
        self.spnApertureValue.setRange(-1e9, 1e9)  # Wide range
        self.spnApertureValue.setSingleStep(0.1)
        layout.addRow("Value:", self.spnApertureValue)

        # Apply button
        self.btnApplyAperture = QPushButton("Apply Aperture Changes")
        layout.addRow(self.btnApplyAperture)

        self.cmbApertureType.currentTextChanged.connect(self.on_aperture_changed)
        self.spnApertureValue.valueChanged.connect(self.on_aperture_changed)
        self.btnApplyAperture.clicked.connect(self.apply_aperture_changes)

    @Slot()
    def load_data(self):
        self.is_loading = True
        optic = self.connector.get_optic()
        if optic and optic.aperture:
            self.cmbApertureType.setCurrentText(optic.aperture.ap_type)
            self.spnApertureValue.setValue(optic.aperture.value)
        else:  # Default or if no aperture set
            self.cmbApertureType.setCurrentIndex(0)  # EPD
            self.spnApertureValue.setValue(10.0)  # Default EPD value
        self.is_loading = False

    @Slot()
    def on_aperture_changed(self):
        # This slot can be used for immediate validation or dynamic UI changes if needed
        # For now, changes are applied via the "Apply" button
        pass

    @Slot()
    def apply_aperture_changes(self):
        if self.is_loading:
            return
        optic = self.connector.get_optic()
        if optic:
            ap_type = self.cmbApertureType.currentText()
            ap_value = self.spnApertureValue.value()
            try:
                # Assuming Optic.set_aperture exists and handles Optic.aperture
                # creation/update
                optic.set_aperture(ap_type, ap_value)
                self.connector.opticChanged.emit()  # Notify that optic has changed
                print(f"Aperture updated: {ap_type}, {ap_value}")
            except (
                ValueError
            ) as e:  # Catch errors from Optic.set_aperture (e.g. telecentric conflict)
                # QMessageBox.warning(self, "Aperture Error", str(e))
                print(f"Aperture Error: {e}")
                self.load_data()  # Revert UI to current optic state


class FieldsEditor(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.is_loading = False

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Field Type
        form_layout = QFormLayout()
        self.cmbFieldType = QComboBox()
        self.cmbFieldType.addItems(["angle", "object_height"])
        form_layout.addRow("Field Type:", self.cmbFieldType)
        main_layout.addLayout(form_layout)

        # Fields Table
        self.tableFields = QTableWidget()
        self.tableFields.setColumnCount(4)
        self.tableFields.setHorizontalHeaderLabels(
            ["X-Field", "Y-Field", "Vignette X", "Vignette Y"]
        )
        self.tableFields.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        main_layout.addWidget(self.tableFields)

        # Buttons
        button_layout = QHBoxLayout()
        self.btnAddField = QPushButton("Add Field")
        self.btnRemoveField = QPushButton("Remove Field")
        self.btnApplyFields = QPushButton("Apply Field Changes")  # For table edits
        button_layout.addWidget(self.btnAddField)
        button_layout.addWidget(self.btnRemoveField)
        button_layout.addWidget(self.btnApplyFields)
        main_layout.addLayout(button_layout)

        self.cmbFieldType.currentTextChanged.connect(
            self.apply_field_type_change
        )  # Apply immediately
        self.btnAddField.clicked.connect(self.add_field)
        self.btnRemoveField.clicked.connect(self.remove_field)
        self.btnApplyFields.clicked.connect(self.apply_table_field_changes)

    @Slot()
    def load_data(self):
        self.is_loading = True
        optic = self.connector.get_optic()
        if optic and optic.field_type:
            self.cmbFieldType.setCurrentText(optic.field_type)

        self.tableFields.setRowCount(0)  # Clear table
        if optic and optic.fields:
            num_fields = optic.fields.num_fields
            self.tableFields.setRowCount(num_fields)
            for i, field_obj in enumerate(optic.fields.fields):
                self.tableFields.setItem(i, 0, QTableWidgetItem(str(field_obj.x)))
                self.tableFields.setItem(i, 1, QTableWidgetItem(str(field_obj.y)))
                self.tableFields.setItem(i, 2, QTableWidgetItem(str(field_obj.vx)))
                self.tableFields.setItem(i, 3, QTableWidgetItem(str(field_obj.vy)))
        self.is_loading = False

    @Slot()
    def apply_field_type_change(self):
        if self.is_loading:
            return
        optic = self.connector.get_optic()
        if optic:
            new_type = self.cmbFieldType.currentText()
            try:
                optic.set_field_type(new_type)
                self.connector.opticChanged.emit()
                print(f"Field type changed to: {new_type}")
            except ValueError as e:
                # QMessageBox.warning(self, "Field Type Error", str(e))
                print(f"Field Type Error: {e}")
                # Revert combo box if change is invalid
                self.cmbFieldType.setCurrentText(optic.field_type)

    @Slot()
    def add_field(self):
        optic = self.connector.get_optic()
        if optic:
            # Add a default field, e.g., (0, max_y_field * 0.5) or just (0,0)
            y_val = (
                optic.fields.max_y_field * 0.5
                if optic.fields.num_fields > 0 and optic.fields.max_y_field > 0
                else 0.0
            )
            if optic.fields.num_fields == 0:
                y_val = 1.0  # Default if no fields yet

            optic.add_field(y=y_val)  # Optiland's default for x, vx, vy will be used
            self.load_data()  # Refresh table
            self.connector.opticChanged.emit()
            print("Field added.")

    @Slot()
    def remove_field(self):
        optic = self.connector.get_optic()
        current_row = self.tableFields.currentRow()
        if optic and current_row != -1 and optic.fields.num_fields > current_row:
            # Optic.fields is a FieldGroup, its `fields` attribute is a list
            del optic.fields.fields[current_row]
            self.load_data()  # Refresh table
            self.connector.opticChanged.emit()
            print(f"Field at row {current_row} removed.")

    @Slot()
    def apply_table_field_changes(self):
        optic = self.connector.get_optic()
        if optic and optic.fields:
            if self.tableFields.rowCount() == optic.fields.num_fields:
                changed = False
                for i in range(self.tableFields.rowCount()):
                    try:
                        x = float(self.tableFields.item(i, 0).text())
                        y = float(self.tableFields.item(i, 1).text())
                        vx = float(self.tableFields.item(i, 2).text())
                        vy = float(self.tableFields.item(i, 3).text())

                        field_obj = optic.fields.fields[i]
                        if (
                            field_obj.x != x
                            or field_obj.y != y
                            or field_obj.vx != vx
                            or field_obj.vy != vy
                        ):
                            field_obj.x, field_obj.y, field_obj.vx, field_obj.vy = (
                                x,
                                y,
                                vx,
                                vy,
                            )
                            changed = True
                    except ValueError:
                        print(f"Invalid data in fields table row {i}")
                        self.load_data()  # Revert to original
                        return
                if changed:
                    self.connector.opticChanged.emit()
                    print("Field table changes applied.")
            else:  # Row count mismatch, full reload
                self.load_data()


class WavelengthsEditor(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.is_loading = False

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.tableWavelengths = QTableWidget()
        self.tableWavelengths.setColumnCount(3)
        self.tableWavelengths.setHorizontalHeaderLabels(
            ["Value (µm)", "Unit", "Primary"]
        )
        self.tableWavelengths.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        # Corrected Line:
        self.tableWavelengths.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )  # Keep row selection
        self.tableWavelengths.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )  # Ensure only one row can be selected

        main_layout.addWidget(self.tableWavelengths)

        button_layout = QHBoxLayout()
        self.btnAddWavelength = QPushButton("Add Wavelength")
        self.btnRemoveWavelength = QPushButton("Remove Wavelength")
        self.btnSetPrimary = QPushButton("Set Selected as Primary")
        self.btnApplyWavelengths = QPushButton("Apply Wavelength Changes")

        button_layout.addWidget(self.btnAddWavelength)
        button_layout.addWidget(self.btnRemoveWavelength)
        button_layout.addWidget(self.btnSetPrimary)
        button_layout.addWidget(self.btnApplyWavelengths)
        main_layout.addLayout(button_layout)

        self.btnAddWavelength.clicked.connect(self.add_wavelength)
        self.btnRemoveWavelength.clicked.connect(self.remove_wavelength)
        self.btnSetPrimary.clicked.connect(self.set_primary_wavelength)
        self.btnApplyWavelengths.clicked.connect(self.apply_table_wavelength_changes)

        # Connect signals for automatic refresh from connector
        self.connector.opticLoaded.connect(self.load_data)
        self.connector.opticChanged.connect(self.load_data)

    @Slot()
    def load_data(self):
        self.is_loading = True
        self.tableWavelengths.setRowCount(0)
        optic = self.connector.get_optic()
        if optic and optic.wavelengths:
            num_wl = optic.wavelengths.num_wavelengths
            self.tableWavelengths.setRowCount(num_wl)
            for i, wl_obj in enumerate(optic.wavelengths.wavelengths):
                # Optiland stores Wavelength.value in um after conversion.
                # The original value and unit are in Wavelength._value, Wavelength._unit
                self.tableWavelengths.setItem(
                    i, 0, QTableWidgetItem(f"{wl_obj.value:.4f}")
                )  # Display value in um
                item_unit = QTableWidgetItem(
                    wl_obj._unit if hasattr(wl_obj, "_unit") else "um"
                )  # Original unit
                item_unit.setFlags(
                    item_unit.flags() & ~Qt.ItemFlag.ItemIsEditable
                )  # Unit not directly editable here
                self.tableWavelengths.setItem(i, 1, item_unit)

                primary_item = QTableWidgetItem("Yes" if wl_obj.is_primary else "No")
                primary_item.setFlags(
                    primary_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                self.tableWavelengths.setItem(i, 2, primary_item)
        self.is_loading = False

    @Slot()
    def add_wavelength(self):
        optic = self.connector.get_optic()
        if optic:
            # Add a default new wavelength, e.g. 0.6328 um (HeNe)
            # It will become primary if it's the only one or if explicitly set
            is_new_primary = optic.wavelengths.num_wavelengths == 0
            optic.add_wavelength(0.6328, is_primary=is_new_primary, unit="um")
            self.load_data()
            self.connector.opticChanged.emit()
            print("Wavelength added.")

    @Slot()
    def remove_wavelength(self):
        optic = self.connector.get_optic()
        current_row = self.tableWavelengths.currentRow()
        if (
            optic
            and current_row != -1
            and optic.wavelengths.num_wavelengths > current_row
        ):
            if optic.wavelengths.num_wavelengths == 1:
                print("Cannot remove the last wavelength.")
                return

            was_primary = optic.wavelengths.wavelengths[current_row].is_primary
            del optic.wavelengths.wavelengths[current_row]

            if was_primary and optic.wavelengths.num_wavelengths > 0:
                # If the removed was primary, make the new first one primary
                optic.wavelengths.wavelengths[0].is_primary = True

            self.load_data()
            self.connector.opticChanged.emit()
            print(f"Wavelength at row {current_row} removed.")

    @Slot()
    def set_primary_wavelength(self):
        optic = self.connector.get_optic()
        current_row = self.tableWavelengths.currentRow()
        if (
            optic
            and current_row != -1
            and optic.wavelengths.num_wavelengths > current_row
        ):
            for i, wl_obj in enumerate(optic.wavelengths.wavelengths):
                wl_obj.is_primary = i == current_row
            self.load_data()
            self.connector.opticChanged.emit()
            print(f"Wavelength at row {current_row} set as primary.")

    @Slot()
    def apply_table_wavelength_changes(self):
        optic = self.connector.get_optic()
        if self.is_loading or not optic or not optic.wavelengths:
            return

        changed = False
        if self.tableWavelengths.rowCount() == optic.wavelengths.num_wavelengths:
            for i in range(self.tableWavelengths.rowCount()):
                try:
                    # Only wavelength value is editable from table for now
                    new_val_um_str = self.tableWavelengths.item(i, 0).text()
                    new_val_um = float(new_val_um_str)

                    wl_obj = optic.wavelengths.wavelengths[i]
                    # Optiland's Wavelength object stores original value and unit.
                    # To change value while keeping original unit:
                    # wl_obj._value = new_val_converted_back_to_original_unit
                    # wl_obj._value_in_um = wl_obj._convert_to_um()
                    # For simplicity, if table shows µm, we modify it assuming µm.
                    # This means original unit info might get less relevant if edited
                    # this way. A better way would be to edit _value and _unit, then
                    # re-calculate .value
                    if wl_obj.value != new_val_um:  # .value is always in um
                        wl_obj._value = (
                            new_val_um  # Assume new value is directly in um for now
                        )
                        wl_obj._unit = "um"  # Mark it as um
                        wl_obj._value_in_um = (
                            new_val_um  # Recalculate (or set directly)
                        )
                        changed = True
                except ValueError:
                    print(f"Invalid numeric data in Wavelengths table row {i + 1}.")
                    self.load_data()  # Revert
                    return
            if changed:
                self.connector.opticChanged.emit()
                print("Wavelength table changes applied.")
        else:
            self.load_data()  # Row count mismatch
