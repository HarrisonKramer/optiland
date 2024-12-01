import QtQuick
import QtQuick.Window
import QtQuick.Controls

Window {
    id: mainWindow
    width: 1000
    height: 580
    minimumWidth: 800
    minimumHeight: 500
    visible: true
    color: "#00000000"
    title: qsTr("Optiland")

    // Load Theme
    property var theme: Theme {}

    // Properties
    property int windowMargin: 10

    Rectangle {
        id: bg
        color: mainWindow.theme.currentTheme.primary
        border.color: mainWindow.theme.currentTheme.accent
        border.width: 1
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.rightMargin: mainWindow.windowMargin
        anchors.leftMargin: mainWindow.windowMargin
        anchors.bottomMargin: mainWindow.windowMargin
        anchors.topMargin: mainWindow.windowMargin
        z: 0

        // Switch to toggle themes
        Switch {
            id: themeSwitch
            anchors.top: parent.top
            anchors.right: parent.right
            anchors.margins: 16
            text: "Dark Mode"
            checked: mainWindow.theme.currentTheme === mainWindow.theme.darkTheme
            onCheckedChanged: {
                mainWindow.theme.currentTheme = checked ? mainWindow.theme.darkTheme : mainWindow.theme.lightTheme
            }
        }
    }
}
