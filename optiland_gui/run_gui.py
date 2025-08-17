"""Entry point for launching the Optiland GUI application.

This script initializes the QApplication and the MainWindow, starting the
event loop to run the graphical user interface for Optiland.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import sys
import time

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

    # splash screen
    splash = QSplashScreen()
    splash.setPixmap(scaled_pixmap)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.setEnabled(False)

    splash.show()
    app.processEvents()

    # Use a loop to simulate a longer loading process and update the message
    # 3 seconds
    total_steps = 30
    for i in range(total_steps + 1):
        message = f"<h3>Initializing application... {i * 100 // total_steps}%</h3>"
        splash.showMessage(
            message,
            Qt.AlignBottom | Qt.AlignHCenter,
            Qt.white,
        )
        time.sleep(0.1)
        app.processEvents()

    window = MainWindow()
    window.show()

    # close the splash screen
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
