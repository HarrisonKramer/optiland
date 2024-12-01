import QtQuick

QtObject {
    id: themeManager

    property var currentTheme: lightTheme

    property var lightTheme: QtObject {
        property color primary: "#DFF2EB"
        property color secondary: "#B9E5E8"
        property color text: "#4A628A"
        property color accent: "#7AB2D3"
    }

    property var darkTheme: QtObject {
        property color primary: "#222831"
        property color secondary: "#31363F"
        property color text: "#EEEEEE"
        property color accent: "#76ABAE"
    }
}
