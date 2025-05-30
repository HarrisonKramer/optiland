# optiland_gui/run_gui.py
import sys
from PySide6.QtWidgets import QApplication
# Use a relative import here because run_gui.py is part of the optiland_gui package
from .main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Optional: Apply a stylesheet (e.g., from qt-material or qdarkstyle)
    # from qt_material import apply_stylesheet
    # apply_stylesheet(app, theme='dark_teal.xml')

    # Or load a custom QSS file:
    # try:
    #     with open("resources/styles.qss", "r") as f:
    #         app.setStyleSheet(f.read())
    # except FileNotFoundError:
    #     print("resources/styles.qss not found. Using default style.")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())