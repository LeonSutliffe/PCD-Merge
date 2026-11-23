"""The Inspect mode's right-hand panel: measure in the 3D view, list and
export the measurements, and save screenshots.

Measuring itself works in every mode (the Measure button above the view);
this panel is where the measurements are looked at in detail. Inspect turns
measuring on when it opens.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QPushButton, QRadioButton,
                               QScrollArea, QVBoxLayout, QWidget)

from ..measure import Measurement, write_csv
from .common import HINT_STYLE
from .edit_tools import MeasureTool

ROLE_ID = Qt.UserRole + 1
SHOT_SCALES = [("As shown", 1), ("2× larger", 2), ("4× larger", 4)]


class InspectPanel(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.doc = win.doc
        self.view = win.view
        self._active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)

        # ---- measure
        g = QGroupBox("Measure")
        gl_ = QVBoxLayout(g)
        row = QHBoxLayout()
        self.rb_distance = QRadioButton("Distance")
        self.rb_distance.setToolTip("Click two points: the distance between them, the "
                                    "horizontal distance, the height difference and the slope.")
        self.rb_point = QRadioButton("Point")
        self.rb_point.setToolTip("Click one point: its X, Y and Z coordinates.")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_distance)
        grp.addButton(self.rb_point)
        self.rb_distance.setChecked(True)
        self.rb_distance.toggled.connect(self._kind_changed)
        row.addWidget(self.rb_distance)
        row.addWidget(self.rb_point)
        row.addStretch(1)
        gl_.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet(HINT_STYLE)
        gl_.addWidget(self.status)
        lay.addWidget(g)

        # ---- the measurements
        g = QGroupBox("Measurements")
        gl_ = QVBoxLayout(g)
        self.list = QListWidget()
        self.list.setToolTip("Tick to show in the view; double-click to rename.")
        self.list.setMinimumHeight(120)
        self.list.itemChanged.connect(self._item_changed)
        self.list.currentItemChanged.connect(self._show_details)
        gl_.addWidget(self.list)
        self.details = QLabel()
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setStyleSheet("font-family: monospace;")
        gl_.addWidget(self.details)
        row = QHBoxLayout()
        self.b_delete = QPushButton("Delete")
        self.b_delete.clicked.connect(self.delete_current)
        self.b_csv = QPushButton("Export CSV…")
        self.b_csv.setToolTip("Save every measurement as a spreadsheet (CSV): points, "
                              "distances, height differences and slopes, in metres.")
        self.b_csv.clicked.connect(self._export_csv)
        row.addWidget(self.b_delete)
        row.addStretch(1)
        row.addWidget(self.b_csv)
        gl_.addLayout(row)
        lay.addWidget(g)

        # ---- screenshot
        g = QGroupBox("Screenshot")
        gl_ = QHBoxLayout(g)
        self.shot_scale = QComboBox()
        for label, _k in SHOT_SCALES:
            self.shot_scale.addItem(label)
        self.shot_scale.setToolTip("Larger screenshots are drawn again at that size, for "
                                   "printing or reports.")
        self.b_shot = QPushButton("Save screenshot…")
        self.b_shot.clicked.connect(self._screenshot_dialog)
        gl_.addWidget(self.shot_scale)
        gl_.addStretch(1)
        gl_.addWidget(self.b_shot)
        lay.addWidget(g)

        note = QLabel("Coming later in 0.11: a viewing file (COPC) that shows every point at "
                      "full quality and opens in CloudCompare, QGIS and web viewers too.")
        note.setWordWrap(True)
        note.setStyleSheet(HINT_STYLE)
        lay.addWidget(note)
        lay.addStretch(1)
        self.refresh()

    # ---------------------------------------------------------------- mode

    def activate(self, on: bool) -> None:
        """Inspect opens with measuring on; leaving it stops measuring."""
        self._active = on
        if on:
            self.start_measuring()
        elif self.measuring():
            self.view.set_tool(None)

    def measuring(self) -> bool:
        return isinstance(self.view.tool, MeasureTool)

    def start_measuring(self) -> None:
        if self.view.view is None or not self.doc.scans:
            return
        if self.doc.aligned and not self.view.aligned_shown():
            self.view.show_aligned()
        if not self.measuring():
            self.view.set_tool(MeasureTool(self.view, self))
        self.measure_status(self._prompt())

    def _prompt(self) -> str:
        if self.measure_kind() == "point":
            return "Click a point in the view to read its coordinates."
        return "Click the first point in the view, then the second."

    def _kind_changed(self, *_):
        if self.measuring():
            self.view.tool.cancel()
            self.view.hint.setText(self.view.tool.hint)
            self.measure_status(self._prompt())

    # ---------------------------------------------------------------- what the tool calls

    def measure_kind(self) -> str:
        return "point" if self.rb_point.isChecked() else "distance"

    def measure_status(self, msg: str) -> None:
        self.status.setText(msg)
        if not self._active:  # the panel is not on show: say it on the status line
            self.win.tasks.set_status(msg)

    def add_measurement(self, kind: str, points) -> Measurement:
        m = Measurement(kind, [list(map(float, p)) for p in points])
        self.doc.add_measurement(m)  # refreshes the list and the view
        self.select(m.id)
        text = "   ".join(m.details())
        self.status.setText(text + "\n\n" + self._prompt())
        self.win.tasks.set_status(("Measured: " if kind == "distance" else "Point: ") + text)
        return m

    # ---------------------------------------------------------------- the list

    def refresh(self) -> None:
        current = self.current_id()
        self.list.blockSignals(True)
        self.list.clear()
        for m in self.doc.measurements:
            it = QListWidgetItem(m.label())
            it.setData(ROLE_ID, m.id)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                        | Qt.ItemIsEditable)
            it.setCheckState(Qt.Checked if m.visible else Qt.Unchecked)
            self.list.addItem(it)
            if m.id == current:
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)
        self._show_details()
        self.update_buttons(self.win.tasks.busy)

    def current_id(self) -> str | None:
        it = self.list.currentItem()
        return it.data(ROLE_ID) if it is not None else None

    def select(self, mid: str) -> None:
        for k in range(self.list.count()):
            if self.list.item(k).data(ROLE_ID) == mid:
                self.list.setCurrentRow(k)
                return

    def _show_details(self, *_):
        m = self.doc.measurement(self.current_id()) if self.current_id() else None
        if m is None:
            self.details.setText("")
            return
        self.details.setText("\n".join(m.details()))

    def _item_changed(self, item):
        mid = item.data(ROLE_ID)
        m = self.doc.measurement(mid)
        if m is None:
            return
        text = item.text().strip()
        if text and text != m.label():
            self.doc.rename_measurement(mid, text)
        self.doc.set_measurement_visible(mid, item.checkState() == Qt.Checked)

    def delete_current(self) -> None:
        mid = self.current_id()
        if mid:
            row = self.list.currentRow()
            self.doc.remove_measurement(mid)
            if self.list.count():
                self.list.setCurrentRow(min(row, self.list.count() - 1))

    def update_buttons(self, busy: bool) -> None:
        self.b_delete.setEnabled(self.current_id() is not None)
        self.b_csv.setEnabled(bool(self.doc.measurements))
        self.b_shot.setEnabled(self.view.view is not None)

    # ---------------------------------------------------------------- saving

    def _export_csv(self):
        start = os.path.join(self.doc.project.folder, "measurements.csv") \
            if self.doc.project.saved else "measurements.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export measurements", start,
                                              "CSV spreadsheet (*.csv)")
        if path:
            self.export_csv(path)

    def export_csv(self, path: str) -> None:
        if not path.lower().endswith(".csv"):
            path += ".csv"
        write_csv(path, self.doc.measurements)
        self.win.tasks.set_status(f"Saved {len(self.doc.measurements)} measurement(s) to {path}")

    def _screenshot_dialog(self):
        start = os.path.join(self.doc.project.folder, "screenshot.png") \
            if self.doc.project.saved else "screenshot.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", start,
                                              "PNG image (*.png);;JPEG image (*.jpg)")
        if path:
            self.save_screenshot(path, SHOT_SCALES[self.shot_scale.currentIndex()][1])

    def save_screenshot(self, path: str, scale: int = 1) -> tuple[int, int]:
        """Save what the 3D view shows; ``scale`` > 1 draws it again larger.
        Returns the image's size."""
        if not os.path.splitext(path)[1]:
            path += ".png"
        img = self.view.screenshot(scale)
        if img is None or img.isNull() or not img.save(path):
            self.win.tasks.set_status(f"The screenshot could not be saved to {path}.")
            return 0, 0
        self.win.tasks.set_status(f"Saved a {img.width()} × {img.height()} screenshot to "
                                  f"{path}")
        return img.width(), img.height()
