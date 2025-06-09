# optiland_gui/widgets/python_terminal.py
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget


class PythonTerminalWidget(QWidget):
    commandExecuted = Signal()

    def __init__(self, parent=None, custom_variables=None, theme="dark"):
        super().__init__(parent)
        self.setObjectName("PythonTerminalWidget")

        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()

        kernel_client = self.kernel_manager.client()
        kernel_client.start_channels()

        shell = self.kernel_manager.kernel.shell
        shell.events.register("post_execute", self.on_command_executed)

        self.jupyter_widget = RichJupyterWidget()
        self.jupyter_widget.kernel_manager = self.kernel_manager
        self.jupyter_widget.kernel_client = kernel_client

        self.injected_variables = (
            custom_variables if custom_variables is not None else {}
        )
        self.set_theme(theme)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.jupyter_widget)

        self.inject_variables()

    def inject_variables(self):
        def refresh_gui():
            print("GUI refresh requested from terminal.")
            if "connector" in self.injected_variables:
                self.injected_variables["connector"].opticChanged.emit()
                print("opticChanged signal emitted.")
            else:
                print("Error: 'connector' not found.")

        self.injected_variables["refresh_gui"] = refresh_gui

        if self.injected_variables:
            self.kernel_manager.kernel.shell.push(self.injected_variables)

    def on_command_executed(self):
        self.commandExecuted.emit()

    def set_theme(self, theme="dark"):
        if theme == "dark":
            self.jupyter_widget.syntax_style = "monokai"
            self.jupyter_widget.style_sheet = """
                RichJupyterWidget {
                    background-color: #2B2B2B;
                    color: #F8F8F2;
                    selection-background-color: #44475A;
                }
            """
        else:
            self.jupyter_widget.syntax_style = "default"
            self.jupyter_widget.style_sheet = """
                RichJupyterWidget {
                    background-color: #FFFFFF;
                    color: #000000;
                    selection-background-color: #AAD5FF;
                }
            """
        self.jupyter_widget.execute(source="", hidden=True)

    def push_variable(self, name: str, variable):
        self.injected_variables[name] = variable
        self.kernel_manager.kernel.shell.push({name: variable})

    def shutdown_kernel(self):
        if self.kernel_manager.is_alive():
            self.kernel_manager.shutdown_kernel()
