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
    property var theme: Theme {
        id: currentTheme
    }

    // Properties
    property int windowMargin: 10
    property color primaryColor: "transparent"
    property color secondaryColor: "transparent"
    property color accentColor: "transparent"
    property color textColor: "transparent"

    Component.onCompleted: {
        primaryColor = theme.currentTheme.primary
        secondaryColor = theme.currentTheme.secondary
        accentColor = theme.currentTheme.accent
        textColor = theme.currentTheme.text
    }

    Rectangle {
        id: bg
        color: mainWindow.primaryColor
        border.color: mainWindow.primaryColor
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

        Rectangle {
            id: appContainer
            color: "#00000000"
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.leftMargin: 1
            anchors.rightMargin: 1
            anchors.topMargin: 1
            anchors.bottomMargin: 1

            Rectangle {
                id: topBar
                height: 60
                color: mainWindow.primaryColor
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.leftMargin: 0
                anchors.rightMargin: 0
                anchors.topMargin: 0
            }

            Rectangle {
                id: content
                color: mainWindow.secondaryColor
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: topBar.bottom
                anchors.bottom: parent.bottom
                anchors.leftMargin: 0
                anchors.rightMargin: 0
                anchors.topMargin: 0
                anchors.bottomMargin: 0
            }
        }
    }
}
