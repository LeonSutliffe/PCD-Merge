"""Edit mode's selection tools: what the mouse does in the 3D view.

Each tool turns mouse input into regions in space (see :mod:`pcd_merge.edits`)
and hands them to the Edit panel, which adds them to the selection. Shift
adds to the selection; without it a new selection replaces the old one.
Alt + drag still rotates the view, and the right and middle buttons and the
wheel pan and zoom as usual.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen, QPolygonF

from ..edits import ScreenShape, Sphere

OUTLINE = QColor(255, 220, 40)


def _adding(ev) -> bool:
    return bool(ev.modifiers() & Qt.ShiftModifier)


class Tool:
    name = ""
    hint = ""
    cursor = Qt.CrossCursor

    def __init__(self, view, panel):
        self.view = view
        self.panel = panel

    def press(self, ev) -> bool:
        return True

    def move(self, ev) -> bool:
        return bool(ev.buttons() & Qt.LeftButton)

    def release(self, ev) -> bool:
        return True

    def leave(self) -> None:
        pass

    def cancel(self) -> None:
        pass

    def paint(self, painter) -> None:
        pass


class BrushTool(Tool):
    """Click a point to select the points around it; drag to paint."""

    name = "brush"
    hint = "Click or drag over points to select them   Shift: add   Alt-drag: rotate"

    def __init__(self, view, panel):
        super().__init__(view, panel)
        self._down = False
        self._last = None
        self._mouse = None

    @property
    def radius(self) -> float:
        return self.panel.brush_radius()

    def _dab(self, ev) -> None:
        hit = self.view.pick(ev.position())
        if hit is None:
            return
        r = self.radius
        if self._last is not None and np.linalg.norm(hit - self._last) < 0.4 * r:
            return  # close to the last dab: nothing new
        self._last = hit
        self.panel.extend_selection(Sphere(hit, r))

    def press(self, ev):
        self._down = True
        self._last = None
        self.panel.begin_selection(add=_adding(ev), kind=self.name)
        self._dab(ev)
        return True

    def move(self, ev):
        self._mouse = ev.position()
        if self._down and ev.buttons() & Qt.LeftButton:
            self._dab(ev)
            return True
        return False  # hovering: just the circle moves

    def release(self, ev):
        self._down = False
        self._last = None
        return True

    def leave(self):
        self._mouse = None

    def cancel(self):
        self._down = False
        self._mouse = None

    def paint(self, painter):
        if self._mouse is None:
            return
        px = self.radius / max(self.view.pixel_size(), 1e-9)
        px = float(min(max(px, 3.0), 400.0))
        painter.setPen(QPen(OUTLINE, 1.2))
        painter.drawEllipse(self._mouse, px, px)


class _ShapeTool(Tool):
    """Draws a shape on the screen; selects everything inside it, through the
    whole depth of the view."""

    def __init__(self, view, panel):
        super().__init__(view, panel)
        self.points: list[QPointF] = []
        self._add = False

    def press(self, ev):
        self._add = _adding(ev)
        self.points = [ev.position()]
        return True

    def release(self, ev):
        poly = self.polygon()
        self.points = []
        if poly is None:
            return True
        ndc = np.array([self.view.ndc(p) for p in poly])
        self.panel.begin_selection(add=self._add, kind=self.name)
        self.panel.extend_selection(ScreenShape(ndc, self.view.camera_mvp()))
        return True

    def cancel(self):
        self.points = []

    def polygon(self):
        raise NotImplementedError


class RectangleTool(_ShapeTool):
    name = "rectangle"
    hint = ("Drag a rectangle: selects everything inside it, through the whole depth   "
            "Shift: add   Alt-drag: rotate")

    def move(self, ev):
        if self.points and ev.buttons() & Qt.LeftButton:
            self.points = [self.points[0], ev.position()]
            return True
        return False

    def polygon(self):
        if len(self.points) < 2:
            return None
        a, b = self.points
        if abs(a.x() - b.x()) < 3 or abs(a.y() - b.y()) < 3:
            return None
        return [QPointF(a.x(), a.y()), QPointF(b.x(), a.y()), QPointF(b.x(), b.y()),
                QPointF(a.x(), b.y())]

    def paint(self, painter):
        if len(self.points) == 2:
            painter.setPen(QPen(OUTLINE, 1.2, Qt.DashLine))
            painter.drawRect(QRectF(self.points[0], self.points[1]).normalized())


class LassoTool(_ShapeTool):
    name = "lasso"
    hint = ("Drag around the points to select: everything inside the loop, through the "
            "whole depth   Shift: add   Alt-drag: rotate")

    def move(self, ev):
        if self.points and ev.buttons() & Qt.LeftButton:
            p = ev.position()
            last = self.points[-1]
            if abs(p.x() - last.x()) + abs(p.y() - last.y()) >= 3:
                self.points.append(p)
            return True
        return False

    def polygon(self):
        return self.points if len(self.points) >= 3 else None

    def paint(self, painter):
        if len(self.points) >= 2:
            painter.setPen(QPen(OUTLINE, 1.2, Qt.DashLine))
            painter.drawPolyline(QPolygonF(self.points + [self.points[0]]))


class GrowTool(Tool):
    """Click an object (a chair, a person, a car): selects the whole of it,
    without the floor or the wall it is against."""

    name = "object"
    hint = ("Click an object to select all of it (not the floor or wall it touches)   "
            "Shift: add   Alt-drag: rotate")
    cursor = Qt.PointingHandCursor

    def press(self, ev):
        hit = self.view.pick(ev.position())
        if hit is not None:
            self.panel.grow_at(hit, add=_adding(ev))
        return True

    def move(self, ev):
        return bool(ev.buttons() & Qt.LeftButton)


class CutTool(Tool):
    """Drag a line across the view: the site (or the chosen part) is split
    in two along the plane through that line and the view direction (in a
    Top view: a vertical wall along the line)."""

    name = "cut"
    hint = "Drag a line to split along it (through the view direction)   Alt-drag: rotate"

    def __init__(self, view, panel):
        super().__init__(view, panel)
        self.points: list[QPointF] = []

    def press(self, ev):
        self.points = [ev.position(), ev.position()]
        return True

    def move(self, ev):
        if self.points and ev.buttons() & Qt.LeftButton:
            self.points[1] = ev.position()
            return True
        return False

    def release(self, ev):
        if len(self.points) == 2:
            a, b = self.points
            if abs(a.x() - b.x()) + abs(a.y() - b.y()) >= 8:
                plane = cut_plane(self.view.ndc(a), self.view.ndc(b), self.view.camera_mvp())
                if plane is not None:
                    self.panel.split_with(plane, ("Side A", "Side B"))
        self.points = []
        return True

    def cancel(self):
        self.points = []

    def paint(self, painter):
        if len(self.points) == 2:
            painter.setPen(QPen(QColor(255, 90, 60), 1.6, Qt.DashLine))
            painter.drawLine(self.points[0], self.points[1])


class MeasureTool(Tool):
    """Click two points for a distance (or one, for its coordinates). The
    points snap to the nearest drawn point under the mouse."""

    name = "measure"
    cursor = Qt.CrossCursor

    def __init__(self, view, panel):
        super().__init__(view, panel)
        self.first = None  # the first point of a distance, project frame
        self._mouse = None

    @property
    def hint(self):
        if self.panel.measure_kind() == "point":
            return "Click a point to read its coordinates   Alt-drag: rotate   Esc: stop"
        return ("Click the first point, then the second, to measure the distance   "
                "Alt-drag: rotate   Esc: start again")

    def press(self, ev):
        if not self.view.measurements_shown():
            self.panel.measure_status("Measurements are made on the aligned scans: choose "
                                      "Show: Aligned above the view.")
            return True
        hit = self.view.pick(ev.position())
        if hit is None:
            self.panel.measure_status("No point there: click on the scan itself.")
            return True
        if self.panel.measure_kind() == "point":
            self.panel.add_measurement("point", [hit])
        elif self.first is None:
            self.first = hit
            self.view.set_pending_point(hit)
            self.panel.measure_status("Now click the second point.")
        else:
            a, self.first = self.first, None
            self.view.set_pending_point(None)
            self.panel.add_measurement("distance", [a, hit])
        return True

    def move(self, ev):
        self._mouse = ev.position()
        return bool(ev.buttons() & Qt.LeftButton)

    def release(self, ev):
        return True

    def leave(self):
        self._mouse = None

    def cancel(self):
        self.first = None
        self._mouse = None
        self.view.set_pending_point(None)

    def paint(self, painter):
        if self.first is None or self._mouse is None:
            return
        mvp = self.view.camera_mvp()
        c = mvp @ np.r_[self.first, 1.0]
        if c[3] <= 1e-9:
            return
        w, h = self.view.view.width(), self.view.view.height()
        x = (c[0] / c[3] + 1) / 2 * w
        y = (1 - c[1] / c[3]) / 2 * h
        painter.setPen(QPen(QColor(90, 220, 255), 1.5, Qt.DashLine))
        painter.drawLine(QPointF(x, y), self._mouse)


class PairTool(Tool):
    """Align by hand: click the same spot on the fixed scan, then on the scan
    to move, and so on. Each click snaps to the nearest drawn point of the
    scan the panel expects next."""

    name = "pairs"
    cursor = Qt.CrossCursor

    @property
    def hint(self):
        return self.panel.pick_hint() + "   Alt-drag: rotate   Esc: stop"

    def press(self, ev):
        i = self.panel.pick_scan()
        hit = self.view.pick(ev.position(), scans=[i], exports=False)
        if hit is None:
            self.panel.pick_missed()
        else:
            self.panel.add_pick(hit)
        self.view.hint.setText(self.hint)
        return True

    def move(self, ev):
        return bool(ev.buttons() & Qt.LeftButton)

    def cancel(self):
        self.panel.stop_picking()


def cut_plane(a_ndc, b_ndc, mvp):
    """The plane through a line drawn on the screen and the view direction,
    as a :class:`~pcd_merge.edits.HalfSpace` (None if it is degenerate)."""
    from ..edits import HalfSpace

    inv = np.linalg.inv(mvp)

    def un(xy, z):
        p = inv @ np.array([xy[0], xy[1], z, 1.0])
        return p[:3] / p[3]

    p0, p1, q0 = un(a_ndc, -0.5), un(a_ndc, 0.5), un(b_ndc, -0.5)
    n = np.cross(p1 - p0, q0 - p0)
    if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
        return None
    n = n / np.linalg.norm(n)
    return HalfSpace(n, float(n @ p0))


TOOLS = {"brush": BrushTool, "rectangle": RectangleTool, "lasso": LassoTool,
         "object": GrowTool, "cut": CutTool}
