"""Right-hand panels of the Edit and Inspect modes.

Placeholders in 0.9: they say what is coming, so the layout and the mode
tabs are in place for 0.10 (Edit) and 0.11 (Inspect) to fill in, each in its
own module, without touching the rest of the window.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .common import HINT_STYLE


class _Placeholder(QWidget):
    TITLE = ""
    TEXT = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        title = QLabel(self.TITLE)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)
        text = QLabel(self.TEXT)
        text.setWordWrap(True)
        text.setTextFormat(Qt.RichText)
        text.setStyleSheet(HINT_STYLE)
        lay.addWidget(text)
        lay.addStretch(1)

    def update_buttons(self, busy: bool):
        pass


class EditPanel(_Placeholder):
    TITLE = "Edit"
    TEXT = ("Clean the scans up and check them: coming in <b>0.10</b>.<ul>"
            "<li>Select by click, brush, grow-to-object, lasso, rectangle or box, in orbit or "
            "walk-through</li>"
            "<li>Delete and restore (every full-resolution point inside the selection), "
            "undo / redo</li>"
            "<li>Split along a line or plane into parts</li>"
            "<li>Automatic clean-up (duplicates, noise, moving objects) moves here</li>"
            "<li>Colour by alignment error, intensity, RGB or height</li></ul>"
            "For now the 3D view, its top/front/side views and the walk-through work here "
            "as in Align.")


class InspectPanel(_Placeholder):
    TITLE = "Inspect"
    TEXT = ("Read-only viewing at full quality: coming in <b>0.11</b>.<ul>"
            "<li>Prepare for inspection: a COPC viewing file that streams every point and "
            "opens in CloudCompare, QGIS and web viewers too</li>"
            "<li>Click-to-measure in 3D: distances, heights, angles, coordinates, saved in the "
            "project and exportable as CSV</li>"
            "<li>Screenshots at a chosen resolution</li></ul>")
