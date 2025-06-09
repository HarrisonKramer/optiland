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
    QSizePolicy,
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

        self.navTree = QTreeWidget()
        self.navTree.setHeaderHidden(True)
        self.navTree.setFixedWidth(150)
        main_layout.addWidget(self.navTree)

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        main_layout.addWidget(self.stackedWidget)

        self.apertureEditor = ApertureEditor(self.connector)
        self.fieldsEditor = FieldsEditor(self.connector)
        self.wavelengthsEditor = WavelengthsEditor(self.connector)

        self.add_nav_item("Aperture", self.apertureEditor)
        self.add_nav_item("Fields", self.fieldsEditor)
        self.add_nav_item("Wavelengths", self.wavelengthsEditor)

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
            placeholder_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            placeholder_layout = QVBoxLayout(placeholder_widget)
            placeholder_layout.addWidget(
                QLabel(f"{placeholder_name} Properties (Placeholder)")
            )
            placeholder_layout.addStretch()
            self.add_nav_item(placeholder_name, placeholder_widget)

        self.navTree.itemClicked.connect(self.on_nav_item_clicked)
        self.navTree.expandAll()
        if self.navTree.topLevelItemCount() > 0:
            self.navTree.setCurrentItem(self.navTree.topLevelItem(0))
            self.stackedWidget.setCurrentIndex(0)

        self.connector.opticLoaded.connect(self.load_properties)
        self.connector.opticChanged.connect(self.load_properties)

    def add_nav_item(self, name, widget):
        item = QTreeWidgetItem(self.navTree, [name])
        index = self.stackedWidget.addWidget(widget)
        item.setData(0, Qt.ItemDataRole.UserRole, index)

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


class PropertyEditorBase(QWidget):
    def __init__(self, connector: OptilandConnector, parent=None):
        super().__init__(parent)
        self.connector = connector
        self.is_loading = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.init_ui()
        self.connector.opticLoaded.connect(self.load_data)
        self.connector.opticChanged.connect(self.load_data)

    def init_ui(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError


class ApertureEditor(PropertyEditorBase):
    def init_ui(self):
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
        self.spnApertureValue.setRange(-1e9, 1e9)
        self.spnApertureValue.setSingleStep(0.1)
        layout.addRow("Value:", self.spnApertureValue)

        self.btnApplyAperture = QPushButton("Apply Aperture Changes")
        layout.addRow(self.btnApplyAperture)

        self.cmbApertureType.currentTextChanged.connect(self.apply_aperture_changes)
        self.spnApertureValue.valueChanged.connect(self.apply_aperture_changes)
        self.btnApplyAperture.clicked.connect(self.apply_aperture_changes)

    @Slot()
    def load_data(self):
        self.is_loading = True
        optic = self.connector.get_optic()
        if optic and optic.aperture:
            self.cmbApertureType.setCurrentText(optic.aperture.ap_type)
            self.spnApertureValue.setValue(optic.aperture.value)
        else:
            self.cmbApertureType.setCurrentIndex(0)
            self.spnApertureValue.setValue(10.0)
        self.is_loading = False

    @Slot()
    def apply_aperture_changes(self):
        if self.is_loading:
            return
        optic = self.connector.get_optic()
        if optic:
            ap_type = self.cmbApertureType.currentText()
            ap_value = self.spnApertureValue.value()
            try:
                optic.set_aperture(ap_type, ap_value)
                self.connector.opticChanged.emit()
                print(f"Aperture updated: {ap_type}, {ap_value}")
            except ValueError as e:
                print(f"Aperture Error: {e}")
                self.load_data()


class FieldsEditor(PropertyEditorBase):
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        form_layout = QFormLayout()
        self.cmbFieldType = QComboBox()
        self.cmbFieldType.addItems(["angle", "object_height"])
        form_layout.addRow("Field Type:", self.cmbFieldType)
        main_layout.addLayout(form_layout)

        self.tableFields = QTableWidget()
        self.tableFields.setColumnCount(4)
        self.tableFields.setHorizontalHeaderLabels(
            ["X-Field", "Y-Field", "Vignette X", "Vignette Y"]
        )
        self.tableFields.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        main_layout.addWidget(self.tableFields)

        button_layout = QHBoxLayout()
        self.btnAddField = QPushButton("Add Field")
        self.btnRemoveField = QPushButton("Remove Field")
        self.btnApplyFields = QPushButton("Apply Field Changes")
        button_layout.addWidget(self.btnAddField)
        button_layout.addWidget(self.btnRemoveField)
        button_layout.addWidget(self.btnApplyFields)
        main_layout.addLayout(button_layout)

        self.cmbFieldType.currentTextChanged.connect(self.apply_field_type_change)
        self.btnAddField.clicked.connect(self.add_field)
        self.btnRemoveField.clicked.connect(self.remove_field)
        self.btnApplyFields.clicked.connect(self.apply_table_field_changes)

    @Slot()
    def load_data(self):
        self.is_loading = True
        optic = self.connector.get_optic()
        if optic and optic.field_type:
            self.cmbFieldType.setCurrentText(optic.field_type)

        self.tableFields.setRowCount(0)
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
                print(f"Field Type Error: {e}")
                self.cmbFieldType.setCurrentText(optic.field_type)

    @Slot()
    def add_field(self):
        optic = self.connector.get_optic()
        if optic:
            y_val = (
                optic.fields.max_y_field * 0.5
                if optic.fields.num_fields > 0 and optic.fields.max_y_field > 0
                else 1.0
                if optic.fields.num_fields == 0
                else 0.0
            )

            optic.add_field(y=y_val)
            self.load_data()
            self.connector.opticChanged.emit()
            print("Field added.")

    @Slot()
    def remove_field(self):
        optic = self.connector.get_optic()
        current_row = self.tableFields.currentRow()
        if optic and current_row != -1 and optic.fields.num_fields > current_row:
            del optic.fields.fields[current_row]
            self.load_data()
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
                    except (ValueError, AttributeError):
                        print(f"Invalid data in fields table row {i}")
                        self.load_data()
                        return
                if changed:
                    self.connector.opticChanged.emit()
                    print("Field table changes applied.")
            else:
                self.load_data()


class WavelengthsEditor(PropertyEditorBase):
    def init_ui(self):
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
        self.tableWavelengths.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.tableWavelengths.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
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

    @Slot()
    def load_data(self):
        self.is_loading = True
        self.tableWavelengths.setRowCount(0)
        optic = self.connector.get_optic()
        if optic and optic.wavelengths:
            num_wl = optic.wavelengths.num_wavelengths
            self.tableWavelengths.setRowCount(num_wl)
            for i, wl_obj in enumerate(optic.wavelengths.wavelengths):
                self.tableWavelengths.setItem(
                    i, 0, QTableWidgetItem(f"{wl_obj.value:.4f}")
                )
                item_unit = QTableWidgetItem(
                    wl_obj._unit if hasattr(wl_obj, "_unit") else "um"
                )
                item_unit.setFlags(item_unit.flags() & ~Qt.ItemFlag.ItemIsEditable)
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
                    new_val_um_str = self.tableWavelengths.item(i, 0).text()
                    new_val_um = float(new_val_um_str)

                    wl_obj = optic.wavelengths.wavelengths[i]
                    if wl_obj.value != new_val_um:
                        wl_obj._value = new_val_um
                        wl_obj._unit = "um"
                        wl_obj._value_in_um = new_val_um
                        changed = True
                except (ValueError, AttributeError):
                    print(f"Invalid numeric data in Wavelengths table row {i + 1}.")
                    self.load_data()
                    return
            if changed:
                self.connector.opticChanged.emit()
                print("Wavelength table changes applied.")
        else:
            self.load_data()
