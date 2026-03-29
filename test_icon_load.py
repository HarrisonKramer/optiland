from __future__ import annotations

import os
import sys

from PySide6.QtGui import QIcon, QPixmap

# Paths from config.py
GUI_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTILAND_ICON_PATH = os.path.join(
    GUI_BASE_DIR, "optiland_gui", "resources", "icons", "optiland_icon.png"
)

print(f"Checking path: {OPTILAND_ICON_PATH}")
print(f"File exists: {os.path.exists(OPTILAND_ICON_PATH)}")

if os.path.exists(OPTILAND_ICON_PATH):
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    pixmap = QPixmap(OPTILAND_ICON_PATH)
    print(f"Pixmap is null: {pixmap.isNull()}")
    print(f"Pixmap size: {pixmap.size()}")
    icon = QIcon(OPTILAND_ICON_PATH)
    print(f"Icon is null: {icon.isNull()}")
