"""The Inspect mode's right-hand panel: a placeholder until 0.11 fills it
in (Edit mode has its own module, :mod:`.edit_panel`)."""

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


class InspectPanel(_Placeholder):
    TITLE = "Inspect"
    TEXT = ("Read-only viewing at full quality: coming in <b>0.11</b>.<ul>"
            "<li>Prepare for inspection: a COPC viewing file that streams every point and "
            "opens in CloudCompare, QGIS and web viewers too</li>"
            "<li>Click-to-measure in 3D: distances, heights, angles, coordinates, saved in the "
            "project and exportable as CSV</li>"
            "<li>Screenshots at a chosen resolution</li></ul>")
