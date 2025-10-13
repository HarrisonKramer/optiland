"""Entry point for launching the Optiland GUI application.

This script initializes the QApplication and the MainWindow, starting the
event loop to run the graphical user interface for Optiland.

Author: Manuel Fragata Mendes, 2025
Refactored by: Jules, 2025
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from .main_window import MainWindow


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    original_pixmap = QPixmap(":/images/logo.png")
    desired_size = QSize(700, 400)
    scaled_pixmap = original_pixmap.scaled(
        desired_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )

    # Create and show splash screen
    splash = QSplashScreen(scaled_pixmap)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.setEnabled(False)
    splash.showMessage(
        "<h3>Initializing application...</h3>",
        Qt.AlignBottom | Qt.AlignHCenter,
        Qt.white,
    )
    splash.show()
    app.processEvents()

    # Initialize the main window while splash is visible. The time taken here
    # is the actual loading time the user experiences.
    window = MainWindow()
    window.show()

    # Close the splash screen once the main window is ready
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
