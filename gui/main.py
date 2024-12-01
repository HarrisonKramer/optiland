# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Load the Theme.qml file as a QQmlComponent
    theme_component = QQmlComponent(engine, QUrl("qml/Theme.qml"))
    theme = theme_component.create()  # Create an instance of the component

    if not theme:
        print("Failed to load Theme.qml:", theme_component.errors())
        sys.exit(-1)

    # Register the Theme object as a global context property
    engine.rootContext().setContextProperty("theme", theme)

    # Load the main QML file
    qml_file = Path(__file__).resolve().parent / "qml/main.qml"
    engine.load(qml_file)

    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
