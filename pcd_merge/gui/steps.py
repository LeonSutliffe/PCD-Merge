"""Building blocks for a panel laid out as numbered steps (Align mode):

- :class:`StepCard`: a numbered card. Done steps get a green tick, the step
  to do next a blue outline, later steps are dimmed.
- :class:`Fold`: "More options ▸", a section that opens on a click.
- :class:`PairList`: one row per pair of overlapping scans, with their
  colours, overlap and fit, and a Good / Fair / Poor label.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from . import theme
from .common import GRADE_COLOURS, cloud_colour

_CARD = {
    "todo": f"QFrame#step {{ background: {theme.BG}; border: 1px solid {theme.LINE};"
            " border-radius: 12px; }",
    "current": "QFrame#step { background: #18202c; border: 1px solid #3b6fb0;"
               " border-radius: 12px; }",
    "ready": f"QFrame#step {{ background: {theme.BG}; border: 1px solid {theme.LINE};"
             " border-radius: 12px; }",
    "done": f"QFrame#step {{ background: {theme.BG}; border: 1px solid {theme.LINE};"
            " border-radius: 12px; }",
}
_NUM = {
    "todo": f"background: {theme.RAISED}; color: {theme.DIM};",
    "current": f"background: {theme.ACCENT}; color: white;",
    "ready": f"background: {theme.RAISED}; color: {theme.TEXT};",
    "done": "background: #3fcf7f; color: #0c2b18;",
}
_GRADE_BG = {"Good": "#183a28", "Fair": "#3d3017", "Poor": "#43201f"}


class StepCard(QFrame):
    def __init__(self, number: int, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("step")
        self.number = number
        self.state = ""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(10)
        self.num = QLabel(str(number))
        self.num.setFixedSize(22, 22)
        self.num.setAlignment(Qt.AlignCenter)
        head.addWidget(self.num)
        self.title = QLabel(title)
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)
        head.addWidget(self.title)
        head.addStretch(1)
        self.note = QLabel()
        self.note.setStyleSheet(f"color: {theme.DIM};")
        head.addWidget(self.note)
        outer.addLayout(head)
        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(8)
        outer.addWidget(self.content)
        self.set_state("todo")

    def set_state(self, state: str, note: str = "") -> None:
        """``todo`` (not yet possible), ``ready`` (possible, not the obvious
        next step), ``current`` (do this next) or ``done``; ``note`` shows on
        the right."""
        self.note.setText(note)
        if note in GRADE_COLOURS:
            self.note.setStyleSheet(f"color: {GRADE_COLOURS[note]}; font-weight: bold;")
        else:
            self.note.setStyleSheet(f"color: {theme.DIM};")
        if state == self.state:
            return
        self.state = state
        self.setStyleSheet(_CARD[state])
        self.num.setText("✓" if state == "done" else str(self.number))
        self.num.setStyleSheet(_NUM[state] + " border-radius: 11px; font-weight: bold;")
        self.title.setStyleSheet(f"color: {theme.DIM};" if state == "todo" else "")


class Fold(QWidget):
    """A link-like toggle that shows or hides ``body``."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.title = title
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setStyleSheet(
            f"QToolButton {{ background: transparent; border: none; color: {theme.ACCENT};"
            " padding: 0; }")
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.toggled.connect(self.set_open)
        lay.addWidget(self.toggle, 0, Qt.AlignLeft)
        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.content)
        self.set_open(False)

    def set_open(self, on: bool) -> None:
        self.toggle.setText(f"{self.title} {'▾' if on else '▸'}")
        self.content.setVisible(on)
        if self.toggle.isChecked() != on:
            self.toggle.setChecked(on)


def _dot(i: int) -> QLabel:
    r, g, b = cloud_colour(i)
    d = QLabel()
    d.setFixedSize(10, 10)
    d.setStyleSheet(f"background: rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)});"
                    " border-radius: 3px;")
    return d


class PairList(QWidget):
    """The quality of each overlapping pair, at a glance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(5)
        self.rows: list[QWidget] = []
        self.hide()

    def show_quality(self, q, names) -> None:
        for w in self.rows:
            w.deleteLater()
        self.rows = []
        pairs = [p for p in (q.pairs if q is not None else []) if p.overlaps]
        for p in pairs[:12]:
            row = QFrame()
            row.setStyleSheet(f"QFrame {{ background: {theme.PANEL}; border-radius: 8px; }}")
            h = QHBoxLayout(row)
            h.setContentsMargins(8, 5, 8, 5)
            h.setSpacing(6)
            h.addWidget(_dot(p.a))
            h.addWidget(_dot(p.b))
            fit = f"{p.rmse * 100:.1f} cm" if p.rmse < 1 else f"{p.rmse:.2f} m"
            info = QLabel(f"overlap {p.overlap:.0%} · fit {fit}")
            info.setStyleSheet(f"color: {theme.DIM}; background: transparent;")
            a = names[p.a] if p.a < len(names) else str(p.a + 1)
            b = names[p.b] if p.b < len(names) else str(p.b + 1)
            row.setToolTip(f"{a} and {b}: {p.overlap:.0%} of their points overlap, fit "
                           f"{fit} RMS, {p.changed:.0%} changed between them (people, "
                           "doors, moved things).")
            h.addWidget(info, 1)
            grade = QLabel(p.grade)
            grade.setStyleSheet(
                f"color: {GRADE_COLOURS.get(p.grade, theme.TEXT)};"
                f" background: {_GRADE_BG.get(p.grade, theme.RAISED)};"
                " border-radius: 8px; padding: 1px 8px; font-weight: bold;")
            h.addWidget(grade)
            self.lay.addWidget(row)
            self.rows.append(row)
        if len(pairs) > 12:
            more = QLabel(f"…and {len(pairs) - 12} more pairs (see the full report).")
            more.setStyleSheet(f"color: {theme.DIM};")
            self.lay.addWidget(more)
            self.rows.append(more)
        self.setVisible(bool(self.rows))
