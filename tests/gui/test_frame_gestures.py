"""Real Qt event routing for native frame handoff and deterministic fallback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication, QMouseEvent, QResizeEvent
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from optiland_gui.widgets.custom_dock_widget import CustomDockWidget
from optiland_gui.widgets.custom_title_bar import CustomTitleBar
from optiland_gui.widgets.frameless_window import FramelessWindow

EDGES = [
    "top_left",
    "top",
    "top_right",
    "left",
    "right",
    "bottom_left",
    "bottom",
    "bottom_right",
]


def test_resize_before_frame_grips_are_initialized_is_safe(qapp):
    class ConstructingFrame(FramelessWindow):
        def __init__(self):
            # Qt's base exists before FramelessWindow builds its edge widgets.
            QMainWindow.__init__(self)

    window = ConstructingFrame()
    try:
        assert not hasattr(window, "_grips")
        assert qapp.sendEvent(window, QResizeEvent(window.size(), window.size()))
        assert window.isHidden()
        assert not hasattr(window, "_grips")
    finally:
        window.close()
        window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def frame(qapp):
    font = Path("C:/Windows/Fonts/segoeui.ttf")
    if not QFontDatabase.families() and font.exists():
        assert QFontDatabase.addApplicationFont(str(font)) >= 0
        qapp.setFont(QFont("Segoe UI", 9))
    window = FramelessWindow()
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    title = CustomTitleBar(QMenuBar(), window)
    window.custom_title_bar_widget = title
    layout.addWidget(title)
    editor = QLineEdit("one two three")
    layout.addWidget(editor)
    layout.addStretch()
    window.setCentralWidget(content)
    title.maximize_restore_requested.connect(
        lambda: window.showNormal() if window.isMaximized() else window.showMaximized()
    )
    window.setMinimumSize(300, 200)
    window.setMaximumSize(1600, 1000)
    window.setGeometry(20, 60, 760, 500)
    window.show()
    window.activateWindow()
    qapp.processEvents()
    yield window
    window._finish_gesture()
    window.close()
    window.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def move_pointer(target, global_point, buttons=Qt.MouseButton.LeftButton):
    QApplication.sendEvent(
        target,
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(target.mapFromGlobal(global_point)),
            QPointF(global_point),
            Qt.MouseButton.NoButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    QApplication.processEvents()


def press_pointer(target, global_point):
    QTest.mousePress(
        target, Qt.MouseButton.LeftButton, pos=target.mapFromGlobal(global_point)
    )
    QApplication.processEvents()


def release_pointer(target, global_point):
    QTest.mouseRelease(
        target, Qt.MouseButton.LeftButton, pos=target.mapFromGlobal(global_point)
    )
    QApplication.processEvents()


def edge_point(window, edge):
    x = (
        2
        if "left" in edge
        else window.width() - 3
        if "right" in edge
        else window.width() // 2
    )
    y = (
        2
        if "top" in edge
        else window.height() - 3
        if "bottom" in edge
        else window.height() // 2
    )
    return window.mapToGlobal(QPoint(x, y))


def title_point(window):
    title = window.custom_title_bar_widget
    return title.mapToGlobal(QPoint(20, title.height() // 2))


def test_public_cursor_helper_respects_disabled_frame_edges(frame):
    left = QPoint(2, frame.height() // 2)
    corner = QPoint(2, 2)
    frame.updateCursorShape(left)
    assert frame.cursor().shape() == Qt.CursorShape.SizeHorCursor
    frame.updateCursorShape(corner)
    assert frame.cursor().shape() == Qt.CursorShape.SizeFDiagCursor
    frame.updateCursorShape(QPoint(100, 100))
    assert frame.cursor().shape() == Qt.CursorShape.ArrowCursor
    frame.setFixedWidth(frame.width())
    QApplication.processEvents()
    frame.updateCursorShape(left)
    assert frame.cursor().shape() == Qt.CursorShape.ArrowCursor
    frame.updateCursorShape(corner)
    assert frame.cursor().shape() == Qt.CursorShape.SizeVerCursor
    frame.showMaximized()
    QApplication.processEvents()
    frame.updateCursorShape(corner)
    assert frame.cursor().shape() == Qt.CursorShape.ArrowCursor


@pytest.mark.parametrize("screen_source", ["current", "primary", "absent"])
def test_snap_without_pointer_screen_uses_available_fallback_or_leaves_geometry(
    frame, monkeypatch, screen_source
):
    available = QRect(-2100, 40, 2000, 1200)
    screen = SimpleNamespace(availableGeometry=lambda: available)
    monkeypatch.setattr(QGuiApplication, "screenAt", lambda _point: None)
    monkeypatch.setattr(
        frame, "screen", lambda: screen if screen_source == "current" else None
    )
    monkeypatch.setattr(
        QGuiApplication,
        "primaryScreen",
        lambda: screen if screen_source == "primary" else None,
    )
    before = frame.geometry()
    frame._snap_fallback(QPoint(available.left(), available.center().y()))
    if screen_source == "absent":
        assert frame.geometry() == before
    else:
        assert frame.geometry() == QRect(-2100, 40, 1000, 1000)
    assert not frame.isMaximized()


@pytest.mark.parametrize("title_state", ["absent", "hidden"])
def test_body_drag_without_visible_title_does_not_move_frame(
    frame, monkeypatch, title_state
):
    title = frame.custom_title_bar_widget
    if title_state == "absent":
        frame.custom_title_bar_widget = None
    else:
        title.hide()
        QApplication.processEvents()
    monkeypatch.setattr(
        frame,
        "_start_system_move",
        lambda: pytest.fail("Body press must not start a window move"),
    )
    before = frame.geometry()
    start = frame.mapToGlobal(QPoint(100, 120))
    press_pointer(frame, start)
    move_pointer(frame, start + QPoint(80, 50))
    release_pointer(frame, start + QPoint(80, 50))
    assert frame.geometry() == before
    assert frame.start_pos is None
    assert QWidget.mouseGrabber() is not frame
    frame.custom_title_bar_widget = title


def test_nested_passive_title_content_keeps_drag_threshold_and_pointer_anchor(
    frame, monkeypatch
):
    title = frame.custom_title_bar_widget
    container = QWidget(title)
    container.setGeometry(title.width() // 2, 8, 80, 24)
    passive = QWidget(container)
    passive.setGeometry(4, 4, 50, 12)
    passive.show()
    container.show()
    QApplication.processEvents()
    start = passive.mapToGlobal(passive.rect().center())
    assert QApplication.widgetAt(start) is passive
    calls = []
    monkeypatch.setattr(
        frame, "_start_system_move", lambda: calls.append(True) or False
    )
    before = frame.geometry()
    press_pointer(passive, start)
    move_pointer(passive, start + QPoint(QApplication.startDragDistance() - 1, 0))
    assert calls == []
    assert frame.geometry() == before
    destination = start + QPoint(45, 35)
    move_pointer(passive, destination)
    assert calls == [True]
    assert frame.geometry() == before.translated(45, 35)
    assert passive.mapFromGlobal(destination) == passive.rect().center()
    release_pointer(passive, destination)
    assert frame.start_pos is None


@pytest.mark.parametrize("next_press", ["editor", "new_edge"])
def test_second_press_replaces_capture_and_uses_current_geometry(
    frame, monkeypatch, next_press
):
    monkeypatch.setattr(frame, "_start_system_resize", lambda edges: False)
    start = edge_point(frame, "right")
    press_pointer(frame, start)
    move_pointer(frame, start + QPoint(20, 0))
    assert QWidget.mouseGrabber() is frame
    before_second = frame.geometry()
    if next_press == "editor":
        editor = frame.findChild(QLineEdit)
        second = editor.mapToGlobal(editor.rect().center())
        press_pointer(editor, second)
        assert frame.start_pos is None
        assert QWidget.mouseGrabber() is not frame
        move_pointer(editor, second + QPoint(20, 0))
        release_pointer(editor, second + QPoint(20, 0))
        assert frame.geometry() == before_second
    else:
        second = edge_point(frame, "bottom")
        press_pointer(frame, second)
        move_pointer(frame, second + QPoint(0, 25))
        assert frame.width() == before_second.width()
        assert frame.height() == before_second.height() + 25
        assert frame.pos() == before_second.topLeft()
        release_pointer(frame, second + QPoint(0, 25))
        assert frame.start_pos is None
        assert QWidget.mouseGrabber() is not frame


def test_non_title_double_click_selects_editor_word_without_maximizing(frame):
    editor = frame.findChild(QLineEdit)
    editor.setCursorPosition(0)
    spy = QSignalSpy(frame.custom_title_bar_widget.maximize_restore_requested)
    before = frame.geometry()
    point = QPoint(
        5 + editor.fontMetrics().horizontalAdvance("one") // 2, editor.height() // 2
    )
    QTest.mouseDClick(editor, Qt.MouseButton.LeftButton, pos=point)
    QApplication.processEvents()
    assert editor.selectedText() == "one"
    assert spy.count() == 0
    assert not frame.isMaximized()
    assert frame.geometry() == before
    assert frame.start_pos is None


@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("motion", ["grow", "minimum", "maximum"])
def test_fallback_resizes_all_edges_with_limits_and_opposite_anchor(
    frame, monkeypatch, edge, motion
):
    monkeypatch.setattr(frame, "_start_system_resize", lambda edges: False)
    before = frame.geometry()
    start = edge_point(frame, edge)
    target = QApplication.widgetAt(start)
    assert target is frame._grips[edge]
    distance = 30 if motion == "grow" else -4000 if motion == "minimum" else 4000
    dx = -distance if "left" in edge else distance if "right" in edge else 0
    dy = -distance if "top" in edge else distance if "bottom" in edge else 0
    destination = start + QPoint(dx, dy)
    minimum = frame.minimumSize().expandedTo(frame.minimumSizeHint())
    expected = QRect(before)
    if dx:
        expected.setWidth(
            max(minimum.width(), min(frame.maximumWidth(), before.width() + distance))
        )
        if "left" in edge:
            expected.moveRight(before.right())
    if dy:
        expected.setHeight(
            max(
                minimum.height(), min(frame.maximumHeight(), before.height() + distance)
            )
        )
        if "top" in edge:
            expected.moveBottom(before.bottom())
    press_pointer(target, start)
    assert frame.cursor().shape() == target.cursor().shape()
    move_pointer(target, destination)
    assert frame.geometry() == expected
    release_pointer(target, destination)
    assert frame.geometry() == expected
    assert not frame.is_resizing
    assert QWidget.mouseGrabber() is not frame
    move_pointer(target, destination + QPoint(40, 40), Qt.MouseButton.NoButton)
    assert frame.geometry() == expected


def test_corner_clamps_each_axis_independently_and_uses_live_layout_minimum(
    frame, monkeypatch
):
    monkeypatch.setattr(frame, "_start_system_resize", lambda edges: False)
    start = edge_point(frame, "bottom_right")
    before = frame.geometry()
    press_pointer(frame._grips["bottom_right"], start)
    frame.centralWidget().setMinimumHeight(450)
    QApplication.processEvents()
    move_pointer(frame, start + QPoint(60, -4000))
    assert frame.width() == before.width() + 60
    assert frame.height() == max(
        frame.minimumHeight(), frame.minimumSizeHint().height()
    )
    assert frame.height() >= 450
    release_pointer(frame, start + QPoint(60, -4000))


@pytest.mark.parametrize("edge", EDGES)
def test_accepted_native_resize_receives_edges_and_has_no_fallback_geometry(
    frame, monkeypatch, edge
):
    calls = []
    monkeypatch.setattr(
        frame, "_start_system_resize", lambda edges: calls.append(edges) or True
    )
    before = frame.geometry()
    start = edge_point(frame, edge)
    press_pointer(frame._grips[edge], start)
    expected = Qt.Edge(0)
    for name, value in [
        ("top", Qt.Edge.TopEdge),
        ("bottom", Qt.Edge.BottomEdge),
        ("left", Qt.Edge.LeftEdge),
        ("right", Qt.Edge.RightEdge),
    ]:
        if name in edge:
            expected |= value
    assert calls == [expected]
    assert QWidget.mouseGrabber() is not frame
    move_pointer(frame, start + QPoint(60, 40))
    assert frame.geometry() == before
    release_pointer(frame, start + QPoint(60, 40))
    assert not frame.is_resizing


@pytest.mark.parametrize("accepted", [False, True])
def test_title_threshold_one_owner_and_native_handoff(frame, monkeypatch, accepted):
    calls = []
    monkeypatch.setattr(
        frame, "_start_system_move", lambda: calls.append(True) or accepted
    )
    title = frame.custom_title_bar_widget
    start = title_point(frame)
    before = frame.geometry()
    press_pointer(title, start)
    move_pointer(title, start + QPoint(QApplication.startDragDistance() - 1, 0))
    assert frame.geometry() == before
    assert calls == []
    destination = start + QPoint(45, 35)
    move_pointer(title, destination)
    assert calls == [True]
    assert frame.geometry() == (before if accepted else before.translated(45, 35))
    release_pointer(title, destination)
    assert not frame.is_moving
    assert frame.start_pos is None


@pytest.mark.parametrize("accepted", [False, True])
def test_maximized_drag_restores_under_proportional_pointer(
    frame, monkeypatch, accepted
):
    normal = frame.geometry()
    frame.showMaximized()
    QApplication.processEvents()
    title = frame.custom_title_bar_widget
    start = title.mapToGlobal(QPoint(title.width() // 2, title.height() // 2))
    press_local = frame.mapFromGlobal(start)
    ratio = press_local.x() / frame.width()
    destination = start + QPoint(40, 60)
    monkeypatch.setattr(frame, "_start_system_move", lambda: accepted)
    press_pointer(title, start)
    move_pointer(title, destination)
    assert not frame.isMaximized()
    assert frame.size() == normal.size()
    assert frame.mapFromGlobal(destination) == QPoint(
        round(ratio * normal.width()), press_local.y()
    )
    release_pointer(frame, destination)


@pytest.mark.parametrize(
    "kind", [QEvent.Type.WindowDeactivate, QEvent.Type.Hide, QEvent.Type.UngrabMouse]
)
@pytest.mark.parametrize("native", [False, True])
def test_capture_loss_cleans_gesture_without_late_motion(
    frame, monkeypatch, kind, native
):
    monkeypatch.setattr(frame, "_start_system_resize", lambda edges: native)
    start = edge_point(frame, "right")
    press_pointer(frame._grips["right"], start)
    QApplication.sendEvent(frame, QEvent(kind))
    before = frame.geometry()
    move_pointer(frame, start + QPoint(100, 0))
    assert frame.geometry() == before
    assert not frame.is_resizing
    assert frame.start_pos is None
    assert QWidget.mouseGrabber() is not frame


def test_missing_release_and_escape_cancel_fallback(frame, monkeypatch):
    monkeypatch.setattr(frame, "_start_system_move", lambda: False)
    title = frame.custom_title_bar_widget
    start = title_point(frame)
    before = frame.geometry()
    press_pointer(title, start)
    move_pointer(title, start + QPoint(40, 40))
    QTest.keyClick(frame, Qt.Key.Key_Escape)
    assert frame.geometry() == before
    assert not frame.is_moving
    press_pointer(title, title_point(frame))
    move_pointer(title, title_point(frame) + QPoint(40, 40))
    after = frame.geometry()
    move_pointer(frame, title_point(frame) + QPoint(100, 100), Qt.MouseButton.NoButton)
    move_pointer(frame, title_point(frame) + QPoint(200, 200))
    assert frame.geometry() == after
    assert not frame.is_moving


def test_escape_from_maximized_drag_preserves_later_restore_geometry(
    frame, monkeypatch
):
    normal = frame.geometry()
    frame.showMaximized()
    QApplication.processEvents()
    title = frame.custom_title_bar_widget
    start = title_point(frame)
    monkeypatch.setattr(frame, "_start_system_move", lambda: False)
    press_pointer(title, start)
    move_pointer(title, start + QPoint(80, 90))
    assert not frame.isMaximized()
    QTest.keyClick(frame, Qt.Key.Key_Escape)
    QApplication.processEvents()
    assert frame.isMaximized()
    assert frame.normalGeometry() == normal
    frame.showNormal()
    QApplication.processEvents()
    assert frame.geometry() == normal


@pytest.mark.parametrize(
    "stage",
    ["pending", "fallback_move", "fallback_resize", "native_move", "native_resize"],
)
@pytest.mark.parametrize("input_kind", ["move", "release", "escape", "text"])
def test_other_window_input_ends_tracking_without_consuming_or_snapping(
    frame, monkeypatch, stage, input_kind
):
    class PointerReceiver(QWidget):
        def __init__(self):
            super().__init__()
            self.moves = 0
            self.releases = 0
            self.keys = []

        def mouseMoveEvent(self, event):
            self.moves += 1

        def mouseReleaseEvent(self, event):
            self.releases += 1

        def keyPressEvent(self, event):
            self.keys.append(event.key())

    receiver = PointerReceiver()
    receiver.setMouseTracking(True)
    receiver.setGeometry(80, 100, 200, 150)
    receiver.show()
    QApplication.processEvents()
    accepted = stage.startswith("native_")
    monkeypatch.setattr(frame, "_start_system_move", lambda: accepted)
    monkeypatch.setattr(frame, "_start_system_resize", lambda edges: accepted)
    monkeypatch.setattr(
        frame,
        "_snap_fallback",
        lambda pos: pytest.fail("Other-window input must not snap the frame"),
    )
    resize = stage.endswith("resize")
    start = edge_point(frame, "right") if resize else title_point(frame)
    press_pointer(frame, start)
    if stage != "pending":
        move_pointer(frame, start + QPoint(40, 40))
    assert frame._gesture == (
        "pending_move" if stage == "pending" else stage.removeprefix("fallback_")
    )
    before = frame.geometry()
    destination = receiver.mapToGlobal(QPoint(30, 30))
    if input_kind == "move":
        move_pointer(receiver, destination)
        assert receiver.moves == 1
    elif input_kind == "release":
        release_pointer(receiver, destination)
        assert receiver.releases == 1
    else:
        key = Qt.Key.Key_Escape if input_kind == "escape" else Qt.Key.Key_T
        QTest.keyClick(receiver, key)
        assert receiver.keys == [key]
    assert frame.geometry() == before
    assert frame.start_pos is None
    assert QWidget.mouseGrabber() is not frame
    move_pointer(frame, start + QPoint(100, 100))
    assert frame.geometry() == before
    receiver.close()
    receiver.deleteLater()


@pytest.mark.parametrize("operation", ["move", "resize"])
def test_owned_native_escape_is_forwarded_without_local_geometry_cancel(
    frame, monkeypatch, operation
):
    class KeyReceiver(QObject):
        def __init__(self):
            super().__init__()
            self.keys = []

        def eventFilter(self, watched, event):
            if event.type() == QEvent.Type.KeyPress:
                self.keys.append(event.key())
            return False

    receiver = KeyReceiver()
    frame.installEventFilter(receiver)
    monkeypatch.setattr(frame, f"_start_system_{operation}", lambda *args: True)
    start = title_point(frame) if operation == "move" else edge_point(frame, "right")
    press_pointer(frame, start)
    if operation == "move":
        move_pointer(frame, start + QPoint(40, 40))
    before = frame.geometry()
    QTest.keyClick(frame, Qt.Key.Key_Escape)
    assert receiver.keys == [Qt.Key.Key_Escape]
    assert frame.geometry() == before
    assert frame.start_pos is None
    frame.removeEventFilter(receiver)


def test_offscreen_native_refusal_uses_real_qwindow_fallback(frame, qapp):
    if qapp.platformName() != "offscreen":
        pytest.skip("Synthetic input must not enter a desktop system move loop")
    start = edge_point(frame, "right")
    before = frame.geometry()
    press_pointer(frame._grips["right"], start)
    assert frame._gesture == "resize"
    move_pointer(frame, start + QPoint(35, 0))
    assert frame.width() == before.width() + 35
    release_pointer(frame, start + QPoint(35, 0))
    before = frame.geometry()
    start = title_point(frame)
    press_pointer(frame.custom_title_bar_widget, start)
    move_pointer(frame, start + QPoint(45, 20))
    assert frame._gesture == "move"
    assert frame.geometry() == before.translated(45, 20)
    release_pointer(frame, start + QPoint(45, 20))


@pytest.mark.parametrize("accepted", [False, True])
def test_public_qwindow_helpers_are_called_and_respect_native_ownership(
    frame, monkeypatch, accepted
):
    calls = []
    handle = frame.windowHandle()
    monkeypatch.setattr(
        handle,
        "startSystemResize",
        lambda edges: calls.append(("resize", edges)) or accepted,
    )
    monkeypatch.setattr(
        handle, "startSystemMove", lambda: calls.append(("move", None)) or accepted
    )
    start = edge_point(frame, "bottom")
    press_pointer(frame._grips["bottom"], start)
    assert calls == [("resize", Qt.Edge.BottomEdge)]
    release_pointer(frame, start)
    start = title_point(frame)
    press_pointer(frame.custom_title_bar_widget, start)
    move_pointer(frame, start + QPoint(30, 20))
    assert calls[-1] == ("move", None)
    assert frame._gesture == ("native_move" if accepted else "move")
    release_pointer(frame, start + QPoint(30, 20))


@pytest.mark.parametrize("operation", ["move", "resize"])
def test_native_helper_can_finish_reentrantly_without_restarting_local_gesture(
    frame, monkeypatch, operation
):
    def native_finishes(*args):
        QApplication.sendEvent(frame, QEvent(QEvent.Type.WindowDeactivate))
        return True

    monkeypatch.setattr(frame, f"_start_system_{operation}", native_finishes)
    start = title_point(frame) if operation == "move" else edge_point(frame, "right")
    press_pointer(frame, start)
    if operation == "move":
        move_pointer(frame, start + QPoint(30, 20))
    before = frame.geometry()
    move_pointer(frame, start + QPoint(60, 40))
    assert frame.geometry() == before
    assert frame.start_pos is None
    assert not frame.is_moving
    assert not frame.is_resizing


def test_double_click_after_pending_press_has_no_stale_move(frame, monkeypatch):
    calls = []
    monkeypatch.setattr(
        frame, "_start_system_move", lambda: calls.append(True) or False
    )
    title = frame.custom_title_bar_widget
    start = title_point(frame)
    press_pointer(title, start)
    QTest.mouseDClick(title, Qt.MouseButton.LeftButton, pos=title.mapFromGlobal(start))
    QApplication.processEvents()
    assert frame.isMaximized()
    assert frame.start_pos is None
    move_pointer(title, start + QPoint(100, 100))
    assert frame.isMaximized()
    assert calls == []


def test_controls_editing_and_floating_docks_do_not_start_frame_gestures(
    frame, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        frame, "_start_system_move", lambda: calls.append("move") or False
    )
    monkeypatch.setattr(
        frame, "_start_system_resize", lambda edges: calls.append("resize") or False
    )
    title = frame.custom_title_bar_widget
    spy = QSignalSpy(title.settings_requested)
    QTest.mouseClick(title.settings_button, Qt.MouseButton.LeftButton)
    assert spy.count() == 1
    before = frame.geometry()
    editor = frame.findChild(QLineEdit)
    editor.setFocus()
    editor.setCursorPosition(len(editor.text()))
    QTest.keyClick(editor, Qt.Key.Key_Left, Qt.KeyboardModifier.ControlModifier)
    assert editor.cursorPosition() < len(editor.text())
    QTest.keyClick(editor, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    assert editor.cursorPosition() == len(editor.text())
    for key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
        QTest.keyClick(editor, key, Qt.KeyboardModifier.ControlModifier)
    assert frame.geometry() == before
    assert not frame.isMaximized()
    dock = CustomDockWidget("Floating", frame)
    dock.setWidget(QWidget())
    dock.setFloating(True)
    dock.show()
    QApplication.processEvents()
    dock_title = dock.titleBarWidget()
    QTest.mouseClick(dock_title, Qt.MouseButton.LeftButton, pos=QPoint(20, 16))
    assert frame.start_pos is None
    assert calls == []
    dock.close()


def test_child_covered_grips_preserve_interior_controls(frame, monkeypatch):
    overlay = QPushButton("Covered content", frame)
    overlay.setGeometry(frame.rect())
    overlay.show()
    QApplication.processEvents()
    assert not overlay.hasMouseTracking()
    point = edge_point(frame, "left")
    target = QApplication.widgetAt(point)
    assert target is frame._grips["left"]
    assert target.cursor().shape() == Qt.CursorShape.SizeHorCursor
    monkeypatch.setattr(frame, "_start_system_resize", lambda edges: False)
    before = frame.geometry()
    press_pointer(target, point)
    move_pointer(target, point - QPoint(40, 0))
    release_pointer(target, point - QPoint(40, 0))
    assert frame.width() == before.width() + 40
    clicked = QSignalSpy(overlay.clicked)
    QTest.mouseClick(overlay, Qt.MouseButton.LeftButton, pos=QPoint(200, 200))
    assert clicked.count() == 1
    assert frame.start_pos is None


def test_fixed_axes_and_maximized_frames_disable_only_relevant_grips(frame):
    frame.setFixedWidth(frame.width())
    QApplication.processEvents()
    assert not frame._grips["left"].isVisible()
    assert not frame._grips["right"].isVisible()
    assert frame._grips["top_left"].cursor().shape() == Qt.CursorShape.SizeVerCursor
    frame.showMaximized()
    QApplication.processEvents()
    assert not any(grip.isVisible() for grip in frame._grips.values())


def test_disabled_boundary_and_covering_title_controls_remain_clickable(frame):
    overlay = QPushButton("Control above title", frame)
    overlay.setGeometry(0, 0, 100, 40)
    overlay.show()
    frame.setFixedSize(frame.size())
    QApplication.processEvents()
    assert not any(grip.isVisible() for grip in frame._grips.values())
    point = frame.mapToGlobal(QPoint(2, 2))
    assert QApplication.widgetAt(point) is overlay
    clicked = QSignalSpy(overlay.clicked)
    QTest.mouseClick(overlay, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    assert clicked.count() == 1
    assert frame.start_pos is None


def test_menu_and_popup_input_stays_with_qt(frame):
    title = frame.custom_title_bar_widget
    menu = title.main_menu_bar.addMenu("File")
    menu.addAction("Test action")
    QApplication.processEvents()
    point = title.main_menu_bar.actionGeometry(menu.menuAction()).center()
    before = frame.geometry()
    QTest.mouseClick(title.main_menu_bar, Qt.MouseButton.LeftButton, pos=point)
    QApplication.processEvents()
    assert menu.isVisible()
    assert frame.start_pos is None
    QTest.keyClick(menu, Qt.Key.Key_Escape)
    QApplication.processEvents()
    assert not menu.isVisible()
    assert frame.geometry() == before


def test_hidden_and_fullscreen_frames_do_not_arm_title_or_resize_gestures(frame):
    title = frame.custom_title_bar_widget
    start = title_point(frame)
    frame.hide()
    press_pointer(title, start)
    assert frame.start_pos is None
    frame.showFullScreen()
    QApplication.processEvents()
    assert not any(grip.isVisible() for grip in frame._grips.values())
    press_pointer(title, title_point(frame))
    move_pointer(title, title_point(frame) + QPoint(80, 90))
    assert frame.isFullScreen()
    assert frame.start_pos is None


def test_long_project_name_clips_without_inflating_frame_resize_minimum(
    frame, monkeypatch
):
    title = frame.custom_title_bar_widget
    baseline = frame.minimumSizeHint().width()
    name = "optical-lens-analysis-" * 30 + ".opds"
    title.set_project_name(name)
    QApplication.processEvents()
    assert title.project_label.toolTip() == name
    assert title.project_label.text() == f"Current Project: {name}"
    assert frame.minimumSizeHint().width() == baseline
    monkeypatch.setattr(frame, "_start_system_resize", lambda edges: False)
    before = frame.geometry()
    start = edge_point(frame, "right")
    press_pointer(frame._grips["right"], start)
    move_pointer(frame, start - QPoint(4000, 0))
    release_pointer(frame, start - QPoint(4000, 0))
    assert frame.width() == max(frame.minimumWidth(), baseline)
    assert frame.x() == before.x()
    for button in (
        title.settings_button,
        title.github_button,
        title.help_button,
        title.minimize_button,
        title.maximize_button,
        title.close_button,
    ):
        assert title.rect().contains(button.geometry())
        assert button.width() == 30
    title.set_project_name("")
    assert title.project_label.toolTip() == "UnnamedProject.opds"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_frame_grips_preserve_actual_theme_pixels(frame, theme):
    path = (
        Path(__file__).parents[2]
        / "optiland_gui/resources/styles"
        / f"{theme}_theme.qss"
    )
    frame.setStyleSheet(path.read_text())
    QApplication.processEvents()
    with_grips = frame.grab().toImage()
    for grip in frame._grips.values():
        grip.hide()
    without_grips = frame.grab().toImage()
    assert with_grips == without_grips


@pytest.mark.parametrize("edge", ["left", "right", "top"])
def test_fallback_snap_uses_pointer_screen_and_available_geometry(
    frame, monkeypatch, edge
):
    available = QRect(-2100, 40, 2000, 1200)
    queried = []
    selected = []
    screen = SimpleNamespace(availableGeometry=lambda: available)

    def screen_at(point):
        queried.append(point)
        return screen

    monkeypatch.setattr(QGuiApplication, "screenAt", screen_at)
    monkeypatch.setattr(frame, "setScreen", selected.append)
    monkeypatch.setattr(frame, "_start_system_move", lambda: False)
    start = title_point(frame)
    destination = (
        QPoint(available.center().x(), available.top())
        if edge == "top"
        else QPoint(
            available.left() if edge == "left" else available.right(),
            available.center().y(),
        )
    )
    press_pointer(frame.custom_title_bar_widget, start)
    move_pointer(frame, destination)
    release_pointer(frame, destination)
    assert queried == [destination]
    if edge == "top":
        assert frame.isMaximized()
        assert selected == [screen]
    else:
        assert not selected
        expected_width = max(
            frame.minimumWidth(),
            frame.minimumSizeHint().width(),
            available.width() // 2,
        )
        expected_x = (
            available.left()
            if edge == "left"
            else available.right() - expected_width + 1
        )
        assert frame.geometry() == QRect(
            expected_x, available.top(), expected_width, frame.maximumHeight()
        )


def test_top_snap_on_current_screen_keeps_original_normal_rectangle(frame):
    before = QRect(frame.geometry())
    screen = frame.screen()
    pointer = QPoint(
        screen.availableGeometry().center().x(), screen.availableGeometry().top()
    )
    frame._snap_fallback(pointer)
    QApplication.processEvents()
    assert frame.isMaximized()
    assert frame.screen() is screen
    frame.showNormal()
    QApplication.processEvents()
    assert frame.geometry() == before
