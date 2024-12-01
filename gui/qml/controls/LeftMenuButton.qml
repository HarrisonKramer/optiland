import QtQuick
import QtQuick.Controls
import Qt5Compat.GraphicalEffects

Button {
    id: btnLeftMenu
    text: qsTr("Left Menu Text")

    property url btnIconSource: "../../resources/icons/home_icon.svg"
    property color btnColorDefault: "#1c1d20"
    property color btnColorMouseOver: "#23272E"
    property color btnColorClicked: "#00a1f1"
    property int iconWidth: 18
    property int iconHeight: 18
    property color activeMenuColor: "#55aaff"
    property color activeMenuColorRight: "#2c313c"
    property bool isActiveMenu: false

    width: 250
    height: 60

    QtObject {
        id: internal

        property var dynamicColor: if(btnLeftMenu.down) {
                                       btnLeftMenu.down ? btnLeftMenu.btnColorClicked : btnLeftMenu.btnColorDefault
                                   } else {
                                       btnLeftMenu.hovered ? btnLeftMenu.btnColorMouseOver : btnLeftMenu.btnColorDefault
                                   }
    }

    implicitWidth: 250
    implicitHeight: 60

    background: Rectangle {
        id: bgBtn
        color: internal.dynamicColor

        Rectangle {
            anchors {
                top: parent.top
                left: parent.left
                bottom: parent.bottom
            }
            color: btnLeftMenu.activeMenuColor
            width: 3
            visible: btnLeftMenu.isActiveMenu
        }

        Rectangle {
            anchors {
                top: parent.top
                right: parent.right
                bottom: parent.bottom
            }
            color: btnLeftMenu.activeMenuColorRight
            width: 5
            visible: btnLeftMenu.isActiveMenu
        }
    }

    contentItem: Item {
        anchors.fill: parent
        id: content

        Image {
            id: iconBtn
            source: btnLeftMenu.btnIconSource
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left
            anchors.leftMargin: 26
            sourceSize.width: btnLeftMenu.iconWidth
            sourceSize.height: btnLeftMenu.iconHeight
            height: btnLeftMenu.iconHeight
            width: btnLeftMenu.iconWidth
            fillMode: Image.PreserveAspectFit
            visible: false
            antialiasing: true
        }

        ColorOverlay {
            anchors.fill: iconBtn
            source: iconBtn
            color: "#ffffff"
            anchors.verticalCenter: parent.verticalCenter
            antialiasing: true
            width: btnLeftMenu.iconWidth
            height: btnLeftMenu.iconHeight
        }

        Text {
            color: "#ffffff"
            text: btnLeftMenu.text
            anchors.verticalCenter: parent.verticalCenter
            font: btnLeftMenu.font
            anchors.left: parent.left
            anchors.leftMargin: 75
        }
    }
}
