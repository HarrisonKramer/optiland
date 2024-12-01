import QtQuick
import QtQuick.Controls

Button{
    id: btnToggle
    property url btnIconSource: "../../resources/icons/menu_icon.svg"
    property color btnColorDefault: "#1c1d20"
    property color btnColorMouseOver: "#23272E"
    property color btnColorClicked: "#00a1f1"

    QtObject{
        id: internal

        property var dynamicColor: if(btnToggle.down){
                                       btnToggle.down ? btnToggle.btnColorClicked : btnToggle.btnColorDefault
                                   } else {
                                       btnToggle.hovered ? btnToggle.btnColorMouseOver : btnToggle.btnColorDefault
                                   }

    }

    implicitWidth: 70
    implicitHeight: 60

    background: Rectangle{
        id: bgBtn
        color: internal.dynamicColor

        Image {
            id: iconBtn
            source: btnToggle.btnIconSource
            anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter
            height: 25
            width: 25
            fillMode: Image.PreserveAspectFit
            visible: false
        }
    }
}
