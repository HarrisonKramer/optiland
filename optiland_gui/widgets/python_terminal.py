from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

class PythonTerminalWidget(QWidget):
    """
    A widget that hosts a fully functional, theme-aware IPython/Jupyter terminal.
    """
    commandExecuted = Signal()
    def __init__(self, parent=None, custom_variables=None, theme='dark'):
        super().__init__(parent)
        self.setObjectName("PythonTerminalWidget")

        # Set up the in-process kernel
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        
        kernel_client = self.kernel_manager.client()
        kernel_client.start_channels()
        
        # Get a reference to the IPython shell
        shell = self.kernel_manager.kernel.shell

        # --- HOOK INTO THE POST-EXECUTE EVENT ---
        # This registers our on_command_executed method to be called after any command.
        shell.events.register('post_execute', self.on_command_executed)

        # Create the RichJupyterWidget
        self.jupyter_widget = RichJupyterWidget()
        self.jupyter_widget.kernel_manager = self.kernel_manager
        self.jupyter_widget.kernel_client = kernel_client
        
        # Inject custom variables into the kernel's namespace
        if custom_variables:
            shell.push(custom_variables)
        
        self.injected_variables = custom_variables if custom_variables is not None else {}
        
        # Apply the initial theme
        self.set_theme(theme)

        # Set up the layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.jupyter_widget)

    def on_command_executed(self):
        """Called automatically after a command runs; emits our Qt signal."""
        self.commandExecuted.emit()

    def set_theme(self, theme='dark'):
        """Sets the color theme of the Jupyter widget."""
        # The RichJupyterWidget uses Pygments styles for syntax highlighting.
        # 'monokai' is a popular dark style, and 'default' is a good light one.
        if theme == 'dark':
            self.jupyter_widget.syntax_style = 'monokai'
            # You can also apply a base stylesheet for the widget itself
            self.jupyter_widget.style_sheet = """
                RichJupyterWidget {
                    background-color: #2B2B2B;
                    color: #F8F8F2;
                    selection-background-color: #44475A;
                }
            """
        else: # Light theme
            self.jupyter_widget.syntax_style = 'default'
            self.jupyter_widget.style_sheet = """
                RichJupyterWidget {
                    background-color: #FFFFFF;
                    color: #000000;
                    selection-background-color: #AAD5FF;
                }
            """
        
        # To make the new style apply, we can re-execute an empty cell
        # This forces the frontend to repaint with the new style
        self.jupyter_widget.execute(source="", hidden=True)

    # Create the refresh function
        def refresh_gui():
            """
            Signals the main application that the optic object may have changed
            and the GUI needs to be updated.
            """
            print("GUI refresh requested from terminal...")
            if 'connector' in self.injected_variables:
                self.injected_variables['connector'].opticChanged.emit()
                print("opticChanged signal emitted.")
            else:
                print("Error: 'connector' not found.")

        # Add the function to the variables to be injected
        self.injected_variables['refresh_gui'] = refresh_gui
        
        if self.injected_variables:
            self.kernel_manager.kernel.shell.push(self.injected_variables)

    def push_variable(self, name: str, variable):
        """Injects a variable into the kernel's namespace."""
        self.kernel_manager.kernel.shell.push({name: variable})

    def shutdown_kernel(self):
        """Gracefully shuts down the kernel."""
        if self.kernel_manager.is_alive():
            self.kernel_manager.shutdown_kernel()