import QtQuick

QtObject {
    id: themeManager

    property var currentTheme: lightTheme

    // https://colorhunt.co/palette/dff2ebb9e5e87ab2d34a628a
    property var lightTheme: QtObject {
        property color primary: "#DFF2EB"
        property color secondary: "#B9E5E8"
        property color text: "#4A628A"
        property color accent: "#7AB2D3"
    }

    // https://colorhunt.co/palette/22283131363f76abaeeeeeee
    property var darkTheme: QtObject {
        property color primary: "#222831"
        property color secondary: "#31363F"
        property color text: "#EEEEEE"
        property color accent: "#76ABAE"
    }
}
