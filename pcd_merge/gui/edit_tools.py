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


TOOLS = {"brush": BrushTool, "rectangle": RectangleTool, "lasso": LassoTool,
         "object": GrowTool}
