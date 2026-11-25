"""The Inspect mode's right-hand panel: measure in the 3D view, list and
export the measurements, and save screenshots.

Measuring itself works in every mode (the Measure button above the view);
this panel is where the measurements are looked at in detail. Inspect turns
measuring on when it opens.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QPushButton, QRadioButton,
                               QScrollArea, QVBoxLayout, QWidget)

from ..measure import Measurement, write_csv
from .common import HINT_STYLE
from .edit_tools import MeasureTool
from .save_dialog import ask_save_path

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

        # ---- viewing file
        g = QGroupBox("Viewing file (COPC)")
        gl_ = QVBoxLayout(g)
        note = QLabel("One file with every point, arranged so viewers load only what is on "
                      "screen: it opens at once and stays sharp close up. Send it to anyone: "
                      "it opens in CloudCompare, QGIS, Potree and viewer.copc.io, and any LAS "
                      "software reads it as a normal LAZ file. Deleted points and the "
                      "clean-up set in Edit are left out, as in a merge.")
        note.setWordWrap(True)
        note.setStyleSheet(HINT_STYLE)
        gl_.addWidget(note)
        self.b_copc = QPushButton("Make viewing file…")
        self.b_copc.clicked.connect(lambda: self.win.align_panel.merge(fmt=".copc.laz"))
        gl_.addWidget(self.b_copc)
        lay.addWidget(g)

        # ---- Revit
        g = QGroupBox("For Revit (via ReCap)")
        gl_ = QVBoxLayout(g)
        note = QLabel("Revit links point clouds only as Autodesk .rcp/.rcs files, which "
                      "only Autodesk software can make. This saves an E57 file (in metres, "
                      "with intensity and colour) that ReCap, which comes with Revit, turns "
                      "into an .rcp in a few clicks. The steps are shown after saving. "
                      "Deleted points and the clean-up set in Edit are left out, as in a "
                      "merge.")
        note.setWordWrap(True)
        note.setStyleSheet(HINT_STYLE)
        gl_.addWidget(note)
        row = QHBoxLayout()
        self.b_revit = QPushButton("Export for Revit…")
        self.b_revit.clicked.connect(lambda: self.win.align_panel.merge(revit=True))
        b_steps = QPushButton("How to open it in Revit")
        b_steps.clicked.connect(self._revit_steps)
        row.addWidget(self.b_revit)
        row.addWidget(b_steps)
        gl_.addLayout(row)
        lay.addWidget(g)

        # ---- browser walk-through
        g = QGroupBox("Browser walk-through")
        gl_ = QVBoxLayout(g)
        note = QLabel("One .html file that opens in any web browser by double-clicking it, "
                      "straight into a walk-through: no app, no internet and nothing to "
                      "install. Clients can walk around, look at an overview and measure. "
                      "It holds a thinned copy of the points (as the app's walk-through "
                      "does), the measurements shown here, and leaves out deleted points.")
        note.setWordWrap(True)
        note.setStyleSheet(HINT_STYLE)
        gl_.addWidget(note)
        from ..webview import DEFAULT_SIZE, SIZES

        self.web_size = QComboBox()
        for name in SIZES:
            self.web_size.addItem(name)
        self.web_size.setCurrentText(DEFAULT_SIZE)
        self.web_size.setToolTip("More points show more detail but make a bigger file that "
                                 "needs a faster computer to open. Phones and tablets: Small.")
        gl_.addWidget(self.web_size)
        self.b_web = QPushButton("Make browser walk-through…")
        self.b_web.clicked.connect(lambda: self.win.make_browser_walk(
            SIZES[self.web_size.currentText()]))
        gl_.addWidget(self.b_web)
        lay.addWidget(g)
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
        doc = self.doc
        self.b_copc.setEnabled(not busy and (doc.result is not None or len(doc.scans) == 1))
        self.b_revit.setEnabled(self.b_copc.isEnabled())
        self.b_web.setEnabled(not busy and bool(doc.scans or doc.project.exports))

    # ---------------------------------------------------------------- saving

    def _name(self, what: str) -> str:
        return f"{self.doc.project.name} {what}" if self.doc.project.saved else what

    def _revit_steps(self):
        from .revit_help import show_revit_steps

        show_revit_steps(self)

    def _export_csv(self):
        path, _ = ask_save_path(self, "Export measurements",
                                self._name("measurements.csv"),
                                "CSV spreadsheet (*.csv)", doc=self.doc)
        if path:
            self.export_csv(path)

    def export_csv(self, path: str) -> None:
        if not path.lower().endswith(".csv"):
            path += ".csv"
        write_csv(path, self.doc.measurements)
        self.win.tasks.set_status(f"Saved {len(self.doc.measurements)} measurement(s) to {path}")

    def _screenshot_dialog(self):
        path, _ = ask_save_path(self, "Save screenshot", self._name("screenshot.png"),
                                "PNG image (*.png);;JPEG image (*.jpg)", doc=self.doc)
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
