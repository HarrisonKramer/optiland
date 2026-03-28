"""Full Optimization Panel for the Optiland GUI.

Provides a dockable panel for configuring and running optical-system
optimisations.  The panel is divided into three tabs (Variables, Operands,
Optimizer), a persistent live-updates section, Run/Stop buttons, and a
scrollable results log.

Author: Manuel Fragata Mendes / Kramer Harrison, 2025
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .optiland_connector import OptilandConnector


# ---------------------------------------------------------------------------
# Helper dialogs
# ---------------------------------------------------------------------------


class AddVariableDialog(QDialog):
    """Small dialog for specifying a new optimization variable.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`.
        surface_index: Pre-selected surface number (default 1).
        suggested_type: Pre-selected variable type key (default ``"radius"``).
        parent: Parent widget.
    """

    def __init__(
        self,
        connector: OptilandConnector,
        surface_index: int = 1,
        suggested_type: str = "radius",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Optimization Variable")
        self.setMinimumWidth(340)
        self._connector = connector

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(8)

        # Surface number
        self.spnSurface = QSpinBox()
        self.spnSurface.setRange(1, 999)
        self.spnSurface.setValue(max(1, surface_index))
        form.addRow("Surface #:", self.spnSurface)

        # Variable type
        self.cmbType = QComboBox()
        for display, key in connector.get_common_variable_types():
            self.cmbType.addItem(display, userData=key)
        idx = self.cmbType.findData(suggested_type)
        if idx >= 0:
            self.cmbType.setCurrentIndex(idx)
        form.addRow("Type:", self.cmbType)

        # Coeff number (shown only for asphere_coeff)
        self.spnCoeff = QSpinBox()
        self.spnCoeff.setRange(0, 20)
        self.spnCoeff.setValue(0)
        self._coeff_label = QLabel("Coeff #:")
        form.addRow(self._coeff_label, self.spnCoeff)
        self._update_coeff_visibility()
        self.cmbType.currentIndexChanged.connect(self._update_coeff_visibility)

        # Min / Max
        self.spnMin = QDoubleSpinBox()
        self.spnMin.setDecimals(4)
        self.spnMin.setRange(-1e9, 1e9)
        self.spnMin.setValue(-1000.0)
        self.spnMin.setSpecialValueText("None")
        form.addRow("Min value:", self.spnMin)

        self.spnMax = QDoubleSpinBox()
        self.spnMax.setDecimals(4)
        self.spnMax.setRange(-1e9, 1e9)
        self.spnMax.setValue(1000.0)
        self.spnMax.setSpecialValueText("None")
        form.addRow("Max value:", self.spnMax)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_coeff_visibility(self) -> None:
        is_coeff = self.cmbType.currentData() == "asphere_coeff"
        self._coeff_label.setVisible(is_coeff)
        self.spnCoeff.setVisible(is_coeff)

    def get_variable_dict(self) -> dict:
        """Return the variable descriptor dict from the current form values.

        Returns:
            A dict suitable for
            :meth:`~optiland_gui.optiland_connector.OptilandConnector.add_optimization_variable`.
        """
        vd: dict = {
            "surface_number": self.spnSurface.value(),
            "type": self.cmbType.currentData(),
            "min_val": self.spnMin.value(),
            "max_val": self.spnMax.value(),
        }
        if vd["type"] == "asphere_coeff":
            vd["coeff_number"] = self.spnCoeff.value()
        return vd


class AddOperandDialog(QDialog):
    """Small dialog for specifying a new optimization operand.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`.
        parent: Parent widget.
    """

    def __init__(
        self,
        connector: OptilandConnector,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Optimization Operand")
        self.setMinimumWidth(380)
        self._connector = connector

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(8)

        # Category
        categories = list(connector.get_operand_categories().keys())
        self.cmbCategory = QComboBox()
        self.cmbCategory.addItems(categories)
        form.addRow("Category:", self.cmbCategory)

        # Type — filtered by category
        self.cmbType = QComboBox()
        form.addRow("Type:", self.cmbType)
        self.cmbCategory.currentTextChanged.connect(self._on_category_changed)
        self._on_category_changed(self.cmbCategory.currentText())

        # Target (equality)
        self.spnTarget = QDoubleSpinBox()
        self.spnTarget.setDecimals(6)
        self.spnTarget.setRange(-1e9, 1e9)
        self.spnTarget.setValue(0.0)
        form.addRow("Target:", self.spnTarget)

        # Weight
        self.spnWeight = QDoubleSpinBox()
        self.spnWeight.setDecimals(4)
        self.spnWeight.setRange(0.0, 1e9)
        self.spnWeight.setValue(1.0)
        form.addRow("Weight:", self.spnWeight)

        # Extra input_data JSON
        self.txtInputData = QLineEdit("{}")
        self.txtInputData.setToolTip(
            "JSON dict of extra parameters (optic added automatically).\n"
            'Example: {"surface_number": 1}'
        )
        form.addRow("Parameters (JSON):", self.txtInputData)
        self.cmbType.currentTextChanged.connect(self._update_default_input_data)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @Slot(str)
    def _on_category_changed(self, category: str) -> None:
        """Repopulate the Type combo when the category changes."""
        self.cmbType.blockSignals(True)
        self.cmbType.clear()
        types = self._connector.get_operand_categories().get(category, [])
        self.cmbType.addItems(types)
        self.cmbType.blockSignals(False)
        if types:
            self._update_default_input_data(types[0])

    @Slot(str)
    def _update_default_input_data(self, op_type: str) -> None:
        """Pre-fill the Parameters field with a sensible default."""
        default = self._connector.get_default_operand_input_data_str(op_type)
        self.txtInputData.setText(default)

    def get_operand_dict(self) -> dict:
        """Return the operand descriptor dict from the current form values.

        Returns:
            A dict suitable for
            :meth:`~optiland_gui.optiland_connector.OptilandConnector.add_optimization_operand`.
        """
        return {
            "type": self.cmbType.currentText(),
            "category": self.cmbCategory.currentText(),
            "target": self.spnTarget.value(),
            "min_val": None,
            "max_val": None,
            "weight": self.spnWeight.value(),
            "input_data_str": self.txtInputData.text(),
        }


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------


class OptimizationPanel(QWidget):
    """Full optimization panel with Variables, Operands, and Optimizer tabs.

    Provides controls for defining variables and operands, selecting and
    configuring an optimizer algorithm, live-update options, Run/Stop buttons,
    and a scrollable results log.

    Signals:
        addVariableRequested (int, str): Emitted when the panel requests a new
            variable (surface_index, suggested_type).  Re-emitted from
            ``connector.requestAddOptimizationVariable``.
    """

    addVariableRequested = Signal(int, str)

    def __init__(self, connector: OptilandConnector, parent: QWidget | None = None):
        super().__init__(parent)
        self.connector = connector
        self.setWindowTitle("Optimization")

        self._iteration_count = 0

        self._init_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        """Build the full panel layout."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        splitter = QSplitter(Qt.Vertical)
        outer.addWidget(splitter, 1)

        # Top half: tabs
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_variables_tab(), "Variables")
        self._tabs.addTab(self._build_operands_tab(), "Operands")
        self._tabs.addTab(self._build_optimizer_tab(), "Optimizer")
        top_layout.addWidget(self._tabs)

        # Live updates
        top_layout.addWidget(self._build_live_updates_widget())

        # Run / Stop
        run_layout = QHBoxLayout()
        self.btnRun = QPushButton("▶  Run")
        self.btnStop = QPushButton("■  Stop")
        self.btnStop.setEnabled(False)
        run_layout.addWidget(self.btnRun)
        run_layout.addWidget(self.btnStop)
        top_layout.addLayout(run_layout)

        splitter.addWidget(top_widget)

        # Bottom half: log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(2)
        log_layout.addWidget(QLabel("Results / Log:"))
        self.txtLog = QTextEdit()
        self.txtLog.setReadOnly(True)
        self.txtLog.setMinimumHeight(80)
        log_layout.addWidget(self.txtLog)
        splitter.addWidget(log_widget)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

    def _build_variables_tab(self) -> QWidget:
        """Build the Variables tab widget."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tblVariables = QTableWidget()
        self.tblVariables.setColumnCount(5)
        self.tblVariables.setHorizontalHeaderLabels(
            ["Surface", "Type", "Current Value", "Min", "Max"]
        )
        self.tblVariables.horizontalHeader().setStretchLastSection(True)
        self.tblVariables.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.tblVariables.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.tblVariables)

        btn_layout = QHBoxLayout()
        self.btnAddVariable = QPushButton("+ Add Variable")
        self.btnRemoveVariable = QPushButton("- Remove Selected")
        btn_layout.addWidget(self.btnAddVariable)
        btn_layout.addWidget(self.btnRemoveVariable)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return w

    def _build_operands_tab(self) -> QWidget:
        """Build the Operands tab widget."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tblOperands = QTableWidget()
        self.tblOperands.setColumnCount(5)
        self.tblOperands.setHorizontalHeaderLabels(
            ["Category", "Type", "Target", "Weight", "Parameters"]
        )
        self.tblOperands.horizontalHeader().setStretchLastSection(True)
        self.tblOperands.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tblOperands.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.tblOperands)

        btn_layout = QHBoxLayout()
        self.btnAddOperand = QPushButton("+ Add Operand")
        self.btnRemoveOperand = QPushButton("- Remove Selected")
        btn_layout.addWidget(self.btnAddOperand)
        btn_layout.addWidget(self.btnRemoveOperand)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return w

    def _build_optimizer_tab(self) -> QWidget:
        """Build the Optimizer tab with algorithm selector and dynamic settings."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        form_top = QFormLayout()
        self.cmbAlgorithm = QComboBox()
        for name, cls in self.connector.get_optimizer_catalog():
            self.cmbAlgorithm.addItem(name, userData=cls)
        form_top.addRow("Algorithm:", self.cmbAlgorithm)
        layout.addLayout(form_top)

        # Scrollable dynamic settings area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self._optimizer_settings_widget = QWidget()
        self._optimizer_settings_layout = QFormLayout(self._optimizer_settings_widget)
        self._optimizer_settings_layout.setSpacing(6)
        scroll.setWidget(self._optimizer_settings_widget)
        layout.addWidget(scroll, 1)

        self.cmbAlgorithm.currentIndexChanged.connect(self._rebuild_optimizer_settings)
        self._rebuild_optimizer_settings()

        return w

    def _build_live_updates_widget(self) -> QGroupBox:
        """Build the Live Updates section."""
        group = QGroupBox("Live Updates")
        layout = QHBoxLayout(group)
        layout.setSpacing(12)

        self.chkLiveViz = QCheckBox("Visualization")
        self.chkLiveVars = QCheckBox("Variable Values")
        layout.addWidget(self.chkLiveViz)
        layout.addWidget(self.chkLiveVars)

        layout.addWidget(QLabel("  Frequency:"))
        self.spnFrequency = QSpinBox()
        self.spnFrequency.setRange(1, 1000)
        self.spnFrequency.setValue(10)
        self.spnFrequency.setSuffix(" iters")
        layout.addWidget(self.spnFrequency)
        layout.addStretch()

        return group

    # ------------------------------------------------------------------
    # Optimizer settings form
    # ------------------------------------------------------------------

    def _rebuild_optimizer_settings(self) -> None:
        """Clear and re-populate the Optimizer settings form."""
        # Remove all rows
        while self._optimizer_settings_layout.rowCount() > 0:
            self._optimizer_settings_layout.removeRow(0)
        self._optimizer_param_edits: dict[str, QLineEdit] = {}

        cls = self.cmbAlgorithm.currentData()
        if cls is None:
            return

        try:
            sig = inspect.signature(cls.optimize)
        except (ValueError, TypeError):
            return

        skip = frozenset({"self", "callback"})
        variadic = frozenset(
            {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        )

        for pname, param in sig.parameters.items():
            if pname in skip or param.kind in variadic:
                continue
            default = (
                param.default if param.default is not inspect.Parameter.empty else ""
            )
            edit = QLineEdit(str(default))
            self._optimizer_param_edits[pname] = edit
            self._optimizer_settings_layout.addRow(f"{pname}:", edit)

    def _collect_optimizer_kwargs(self) -> dict:
        """Parse the dynamic settings form and return kwargs for optimize().

        Returns:
            A dict of ``param_name → parsed_value``.
        """
        kwargs: dict = {}
        for pname, edit in getattr(self, "_optimizer_param_edits", {}).items():
            text = edit.text().strip()
            if not text:
                continue
            try:
                # Try int first, then float, then string
                if "." in text or "e" in text.lower():
                    kwargs[pname] = float(text)
                else:
                    kwargs[pname] = int(text)
            except ValueError:
                kwargs[pname] = text
        return kwargs

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire all internal signals."""
        self.btnAddVariable.clicked.connect(self._on_add_variable)
        self.btnRemoveVariable.clicked.connect(self._on_remove_variable)
        self.btnAddOperand.clicked.connect(self._on_add_operand)
        self.btnRemoveOperand.clicked.connect(self._on_remove_operand)
        self.btnRun.clicked.connect(self._on_run)
        self.btnStop.clicked.connect(self._on_stop)

        # Connector signal → open add-variable dialog pre-filled
        self.connector.requestAddOptimizationVariable.connect(
            self.open_add_variable_dialog
        )
        # Reflect variable changes in the lens editor highlighting
        self.connector.optimizationVariablesChanged.connect(
            self._refresh_variables_table
        )

    # ------------------------------------------------------------------
    # Slots — Variables
    # ------------------------------------------------------------------

    @Slot()
    def _on_add_variable(self) -> None:
        """Open the Add Variable dialog from the panel button."""
        self.open_add_variable_dialog(surface_index=1, suggested_type="radius")

    @Slot(int, str)
    def open_add_variable_dialog(
        self, surface_index: int = 1, suggested_type: str = "radius"
    ) -> None:
        """Open the AddVariableDialog pre-filled with the given values.

        Args:
            surface_index: Pre-selected surface number.
            suggested_type: Pre-selected variable type key.
        """
        dlg = AddVariableDialog(
            self.connector, surface_index, suggested_type, parent=self
        )
        if dlg.exec() == QDialog.Accepted:
            self.connector.add_optimization_variable(dlg.get_variable_dict())
            self._refresh_variables_table()

    @Slot()
    def _on_remove_variable(self) -> None:
        row = self.tblVariables.currentRow()
        if row < 0:
            return
        self.connector.remove_optimization_variable(row)
        self._refresh_variables_table()

    @Slot()
    def _refresh_variables_table(self) -> None:
        """Reload the Variables table from the connector."""
        variables = self.connector.get_optimization_variables()
        self.tblVariables.setRowCount(len(variables))
        for i, vd in enumerate(variables):
            cur_val = self.connector.get_variable_current_value(vd)
            cur_str = f"{cur_val:.4f}" if cur_val is not None else "N/A"
            coeff_str = (
                f" [{vd['coeff_number']}]" if vd.get("coeff_number") is not None else ""
            )
            self.tblVariables.setItem(
                i, 0, QTableWidgetItem(str(vd.get("surface_number", "")))
            )
            self.tblVariables.setItem(
                i, 1, QTableWidgetItem(vd.get("type", "") + coeff_str)
            )
            self.tblVariables.setItem(i, 2, QTableWidgetItem(cur_str))
            self.tblVariables.setItem(
                i, 3, QTableWidgetItem(str(vd.get("min_val", "")))
            )
            self.tblVariables.setItem(
                i, 4, QTableWidgetItem(str(vd.get("max_val", "")))
            )

    def _refresh_variables_current_values(self) -> None:
        """Update only the Current Value column (cheaper than full refresh)."""
        variables = self.connector.get_optimization_variables()
        for i, vd in enumerate(variables):
            if i >= self.tblVariables.rowCount():
                break
            cur_val = self.connector.get_variable_current_value(vd)
            cur_str = f"{cur_val:.4f}" if cur_val is not None else "N/A"
            self.tblVariables.setItem(i, 2, QTableWidgetItem(cur_str))

    # ------------------------------------------------------------------
    # Slots — Operands
    # ------------------------------------------------------------------

    @Slot()
    def _on_add_operand(self) -> None:
        dlg = AddOperandDialog(self.connector, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.connector.add_optimization_operand(dlg.get_operand_dict())
            self._refresh_operands_table()

    @Slot()
    def _on_remove_operand(self) -> None:
        row = self.tblOperands.currentRow()
        if row < 0:
            return
        self.connector.remove_optimization_operand(row)
        self._refresh_operands_table()

    @Slot()
    def _refresh_operands_table(self) -> None:
        """Reload the Operands table from the connector."""
        operands = self.connector.get_optimization_operands()
        self.tblOperands.setRowCount(len(operands))
        for i, od in enumerate(operands):
            target_str = f"{od['target']:.4f}" if od.get("target") is not None else "—"
            self.tblOperands.setItem(i, 0, QTableWidgetItem(od.get("category", "")))
            self.tblOperands.setItem(i, 1, QTableWidgetItem(od.get("type", "")))
            self.tblOperands.setItem(i, 2, QTableWidgetItem(target_str))
            self.tblOperands.setItem(i, 3, QTableWidgetItem(str(od.get("weight", 1.0))))
            self.tblOperands.setItem(
                i, 4, QTableWidgetItem(od.get("input_data_str", "{}"))
            )

    # ------------------------------------------------------------------
    # Slots — Run / Stop
    # ------------------------------------------------------------------

    @Slot()
    def _on_run(self) -> None:
        if self.connector.is_optimization_running():
            return

        self.txtLog.clear()
        self._iteration_count = 0
        self.txtLog.append("Starting optimization...")
        self.btnRun.setEnabled(False)
        self.btnStop.setEnabled(True)

        cls = self.cmbAlgorithm.currentData()
        if cls is None:
            self.txtLog.append("Error: no optimizer selected.")
            self.btnRun.setEnabled(True)
            self.btnStop.setEnabled(False)
            return

        optimizer_kwargs = self._collect_optimizer_kwargs()
        freq = self.spnFrequency.value()
        live_viz = self.chkLiveViz.isChecked()
        live_vars = self.chkLiveVars.isChecked()

        def on_progress(n: int) -> None:
            self._iteration_count = n
            if n % freq == 0:
                if live_viz:
                    self.connector.opticChanged.emit()
                if live_vars:
                    self._refresh_variables_current_values()

        def on_finished(summary: str) -> None:
            self.txtLog.append(summary)
            self.btnRun.setEnabled(True)
            self.btnStop.setEnabled(False)
            self.connector.opticChanged.emit()
            self._refresh_variables_table()

        def on_error(message: str) -> None:
            self.txtLog.append(f"Error: {message}")
            self.btnRun.setEnabled(True)
            self.btnStop.setEnabled(False)

        self.connector.run_optimization(
            cls, optimizer_kwargs, on_progress, on_finished, on_error
        )

    @Slot()
    def _on_stop(self) -> None:
        self.connector.stop_optimization()
        self.txtLog.append(
            f"Stop requested. Iterations so far: {self._iteration_count}"
        )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def update_theme(self, theme_name: str) -> None:
        """Propagate theme changes.  Reserved for future embedded plots.

        Args:
            theme_name: ``"dark"`` or ``"light"``.
        """
