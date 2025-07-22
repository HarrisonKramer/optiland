"""Entry point for launching the Optiland GUI application.

This script initializes the QApplication and the MainWindow, starting the
event loop to run the graphical user interface for Optiland.

Author: Manuel Fragata Mendes, 2025
"""

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main():
    """Initializes and runs the Optiland GUI application.

    This function sets up the Qt application environment, creates the main
    window of the application, shows it, and starts the Qt event loop.
    """
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
