"""Controls that float on the 3D view itself, so the hand stays near the
points being worked on:

- the tool rail down the left edge: one button per tool, with its key
  (Move view V, Brush B, Rectangle R, Lasso L, Object O, Cut line X, and
  Measure M in every mode);
- the selection bar along the bottom: "1,635 points selected", Delete,
  Restore and Clear, shown while Edit mode has a selection.

Both are child widgets of the orbit view and move with it when it is
resized; the walk-through has its own help line instead.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout

from . import theme

RAIL_STYLE = f"""
QFrame#rail, QFrame#selbar {{ background: rgba(29, 32, 39, 235); border: 1px solid {theme.LINE};
                              border-radius: 10px; }}
QFrame#rail QToolButton {{ background: transparent; border: none; border-radius: 8px;
                           padding: 0; }}
QFrame#rail QToolButton:hover {{ background: {theme.RAISED}; }}
QFrame#rail QToolButton:checked {{ background: {theme.ACCENT}; }}
QFrame#rail QToolButton:disabled {{ background: transparent; }}
QFrame#rail QFrame#sep {{ background: {theme.LINE}; max-height: 1px; min-height: 1px;
                          margin: 2px 6px; }}
QFrame#selbar QLabel {{ background: transparent; padding: 0 6px; }}
"""

ICON = 40  # button size


def _icon(kind: str, key: str) -> QIcon:
    """A small line drawing for each tool, with its key in the corner."""
    icon = QIcon()
    for mode, col in ((QIcon.Normal, theme.DIM), (QIcon.Active, theme.TEXT),
                      (QIcon.Disabled, theme.FAINT)):
        for state, c in ((QIcon.Off, col), (QIcon.On, "#ffffff")):
            icon.addPixmap(_draw(kind, key, c if mode != QIcon.Disabled else col), mode, state)
    return icon


def _draw(kind: str, key: str, colour: str) -> QPixmap:
    s, scale = ICON, 2
    pm = QPixmap(s * scale, s * scale)
    pm.setDevicePixelRatio(scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(colour), 1.8)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    c = QPointF(s / 2 - 1, s / 2 - 1)
    if kind == "move":  # four arrows
        for ang in range(0, 360, 90):
            a = math.radians(ang)
            tip = c + QPointF(9 * math.cos(a), 9 * math.sin(a))
            p.drawLine(c, tip)
            for d in (-0.6, 0.6):
                p.drawLine(tip, tip - QPointF(4 * math.cos(a + d), 4 * math.sin(a + d)))
    elif kind == "brush":
        p.drawEllipse(c, 9, 9)
        p.setBrush(QColor(colour))
        p.drawEllipse(c, 3, 3)
    elif kind == "rectangle":
        pen.setDashPattern([2.5, 2])
        p.setPen(pen)
        p.drawRect(QRectF(c.x() - 10, c.y() - 7, 20, 14))
    elif kind == "lasso":
        path = QPainterPath(c + QPointF(-9, 1))
        path.cubicTo(c + QPointF(-10, -9), c + QPointF(10, -10), c + QPointF(9, -1))
        path.cubicTo(c + QPointF(8, 7), c + QPointF(-4, 8), c + QPointF(-7, 3))
        path.quadTo(c + QPointF(-9, 9), c + QPointF(-4, 11))
        p.drawPath(path)
    elif kind == "object":  # a cube
        t = [c + QPointF(x, y) for x, y in ((0, -10), (9, -5), (0, 0), (-9, -5))]
        b = [q + QPointF(0, 10) for q in t]
        for a, b_ in ((t[0], t[1]), (t[1], t[2]), (t[2], t[3]), (t[3], t[0]),
                      (t[1], b[1]), (t[2], b[2]), (t[3], b[3]), (b[1], b[2]), (b[2], b[3])):
            p.drawLine(a, b_)
    elif kind == "cut":  # a dashed line with scissors' handles
        pen.setDashPattern([2.5, 2])
        p.setPen(pen)
        p.drawLine(c + QPointF(-10, 10), c + QPointF(10, -10))
        pen.setDashPattern([])
        p.setPen(pen)
        p.drawEllipse(c + QPointF(-7, -5), 3.2, 3.2)
        p.drawEllipse(c + QPointF(5, 7), 3.2, 3.2)
    elif kind == "measure":  # a ruler
        p.save()
        p.translate(c)
        p.rotate(-35)
        p.drawRect(QRectF(-12, -4, 24, 8))
        for i in range(-8, 10, 4):
            p.drawLine(QPointF(i, -4), QPointF(i, -1 if i % 8 else 0.5))
        p.restore()
    f = p.font()
    f.setPixelSize(9)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(colour))
    p.drawText(QRectF(0, 0, s - 3, s - 2), Qt.AlignRight | Qt.AlignBottom, key)
    p.end()
    return pm


class ToolRail(QFrame):
    """The tool buttons down the left edge of the 3D view.

    ``win`` is the main window: pressing a button does what the tool's key
    does (``win._pick_tool``), and Measure presses the view's Measure."""

    def __init__(self, win, host):
        super().__init__(host)
        self.win = win
        self.setObjectName("rail")
        self.setStyleSheet(RAIL_STYLE)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        from .edit_panel import TOOL_KEYS, TOOL_LABELS

        self.buttons: dict = {}
        self.edit_buttons = []
        kinds = {None: "move", "brush": "brush", "rectangle": "rectangle", "lasso": "lasso",
                 "object": "object", "cut": "cut"}
        for key, label, tip in TOOL_LABELS:
            b = self._button(kinds.get(key, "move"), TOOL_KEYS[key],
                             f"{label} ({TOOL_KEYS[key]})\n{tip}")
            b.clicked.connect(lambda _=False, k=key: self._tool(k))
            lay.addWidget(b)
            self.buttons[key] = b
            self.edit_buttons.append(b)
        self.sep = QFrame()
        self.sep.setObjectName("sep")
        lay.addWidget(self.sep)
        self.b_measure = self._button(
            "measure", "M", "Measure (M)\nClick two points for a distance, or one for its "
                            "coordinates. Works in every mode.")
        self.b_measure.clicked.connect(self._measure)
        lay.addWidget(self.b_measure)
        host.installEventFilter(self)
        self.sync()

    def _button(self, kind, key, tip):
        b = QToolButton()
        b.setCheckable(True)
        b.setIcon(_icon(kind, key))
        b.setIconSize(QSize(ICON, ICON))
        b.setFixedSize(ICON, ICON)
        b.setToolTip(tip)
        b.setFocusPolicy(Qt.NoFocus)
        return b

    def _tool(self, key):
        win = self.win
        editing = win.modes.currentIndex() == 1
        if key is None:  # just move the view: no tool, no measuring
            if editing:
                win.edit_panel.choose_tool(None)
            if win.view.b_measure.isChecked():
                win.view.b_measure.setChecked(False)
        else:
            if editing and win.view.b_measure.isChecked() \
                    and win.edit_panel.current_tool() == key:
                # measuring had taken over from this tool: pick it again, not off
                win.edit_panel.choose_tool(None)
            win._pick_tool(key)
        self.sync()

    def _measure(self):
        self.win.view.b_measure.click()
        self.sync()

    def sync(self, *_):
        """Show the tools that fit the mode, and which one is on."""
        win = self.win
        editing = win.modes.currentIndex() == 1
        for b in self.edit_buttons:
            b.setVisible(editing)
        self.sep.setVisible(editing)
        measuring = win.view.b_measure.isChecked()
        current = win.edit_panel.current_tool() if editing else None
        for key, b in self.buttons.items():
            b.setChecked(editing and not measuring and key == current)
            b.setEnabled(bool(win.doc.scans) or key is None)
        self.b_measure.setChecked(measuring)
        self.b_measure.setEnabled(win.view.b_measure.isEnabled())
        self.adjustSize()
        self._place()

    def _place(self):
        host = self.parentWidget()
        if host is None:
            return
        self.move(12, max(8, (host.height() - self.height()) // 2))
        self.raise_()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Resize:
            self._place()
        return False


class SelectionBar(QFrame):
    """'N points selected' with Delete, Restore and Clear, floating at the
    bottom of the 3D view while Edit mode has a selection."""

    def __init__(self, win, host):
        super().__init__(host)
        self.win = win
        self.setObjectName("selbar")
        self.setStyleSheet(RAIL_STYLE)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 6, 6)
        lay.setSpacing(6)
        self.label = QLabel()
        lay.addWidget(self.label)
        self.b_delete = QPushButton("Delete   Del")
        self.b_delete.setProperty("danger", True)
        self.b_delete.setToolTip("Delete the selected points (Del). Undo with Ctrl+Z.")
        self.b_delete.clicked.connect(win.edit_panel.delete_selection)
        self.b_restore = QPushButton("Restore")
        self.b_restore.setToolTip("Bring back deleted points inside the selection.")
        self.b_restore.clicked.connect(win.edit_panel.restore_selection)
        self.b_clear = QPushButton("Clear   Ctrl+D")
        self.b_clear.setToolTip("Select nothing (Ctrl+D).")
        self.b_clear.clicked.connect(win.edit_panel.clear_selection)
        for b in (self.b_delete, self.b_restore, self.b_clear):
            b.setFocusPolicy(Qt.NoFocus)
            lay.addWidget(b)
        host.installEventFilter(self)
        self.hide()

    def sync(self, *_):
        win = self.win
        editing = win.modes.currentIndex() == 1
        if not (editing and win.edit_panel.regions):
            self.hide()
            return
        shown, full = win.edit_panel.selected_estimate()
        self.label.setText(f"<b>{full:,}</b> points selected")
        busy = win.tasks.busy
        self.b_delete.setEnabled(not busy)
        self.b_restore.setEnabled(not busy)
        self.adjustSize()
        self._place()
        self.show()

    def _place(self):
        host = self.parentWidget()
        if host is None:
            return
        self.move((host.width() - self.width()) // 2, host.height() - self.height() - 14)
        self.raise_()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Resize:
            self._place()
        return False

