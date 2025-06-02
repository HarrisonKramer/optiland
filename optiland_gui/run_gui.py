# optiland_gui/run_gui.py
import sys

from PySide6.QtWidgets import QApplication

# Use a relative import here because run_gui.py is part of the optiland_gui package
from .main_window import MainWindow
from .resources import resources_rc

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
