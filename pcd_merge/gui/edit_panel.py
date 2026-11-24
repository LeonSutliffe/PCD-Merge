"""The Edit mode's right-hand panel: select points in the 3D view, delete or
restore them, undo and redo, and save the cleaned cloud.

The selection is a list of regions in space (brush dabs, lassos,
rectangles, grown objects). Delete and Restore turn it into one edit of the
project; the edit applies to every full-resolution point inside it when
the cloud is saved, not only to the points drawn in the view.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QPushButton, QRadioButton,
                               QScrollArea, QVBoxLayout, QWidget)

from ..edits import Edit, HalfSpace, grow_object, union
from .common import HINT_STYLE
from .edit_tools import TOOLS

TOOL_LABELS = [
    (None, "Move the view", "Rotate, pan and zoom; no selecting."),
    ("brush", "Brush", "Click a point to select the points around it, or drag to paint over "
                       "them. The brush size is set below."),
    ("rectangle", "Rectangle", "Drag a rectangle: selects every point inside it, through the "
                               "whole depth of the view (in a Top view: a box from the ground up)."),
    ("lasso", "Lasso", "Draw a loop around the points: selects every point inside it, through "
                       "the whole depth of the view."),
    ("object", "Object", "Click an object such as a chair, a person or a car: selects all of it, "
                         "without the floor or wall it touches."),
    ("cut", "Cut line", "Drag a line across the view to split along it. The cut goes "
                        "through the view direction: in a Top view it is a vertical wall "
                        "along the line."),
]


TOOL_KEYS = {None: "V", "brush": "B", "rectangle": "R", "lasso": "L", "object": "O", "cut": "X"}


class EditPanel(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.doc = win.doc
        self.view = win.view
        self.regions = []  # the current selection
        self.kinds: list[str] = []  # tools that made it (for the edit's label)
        self._active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)
        self.body = QWidget()
        scroll.setWidget(self.body)
        lay = QVBoxLayout(self.body)

        self.align_note = QLabel(
            "The scans are not aligned yet. Edits stay where they are made, so align first: "
            "aligning afterwards moves the scans away from their edits.")
        self.align_note.setWordWrap(True)
        self.align_note.setStyleSheet("color: #d08a00;")
        lay.addWidget(self.align_note)

        # ---- tools
        g = QGroupBox("Select")
        gl_ = QVBoxLayout(g)
        self.tool_group = QButtonGroup(self)
        self.tool_buttons = {}
        row = None
        for k, (key, label, tip) in enumerate(TOOL_LABELS):
            if key == "cut":  # shown with the split options below
                b = QRadioButton(f"{label} ({TOOL_KEYS[key]})")
                b.setToolTip(f"{tip} Key: {TOOL_KEYS[key]} (again to turn it off).")
                self.tool_group.addButton(b, k)
                self.tool_buttons[key] = b
                continue
            if k % 3 == 0:
                row = QHBoxLayout()
                gl_.addLayout(row)
            b = QRadioButton(f"{label} ({TOOL_KEYS[key]})")
            b.setToolTip(f"{tip} Key: {TOOL_KEYS[key]} (again to turn it off).")
            self.tool_group.addButton(b, k)
            self.tool_buttons[key] = b
            row.addWidget(b)
        row.addStretch(1)
        self.tool_buttons[None].setChecked(True)
        self.tool_group.idToggled.connect(self._tool_toggled)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Brush size:"))
        self.brush = QDoubleSpinBox()
        self.brush.setRange(0.01, 20.0)
        self.brush.setDecimals(2)
        self.brush.setSingleStep(0.05)
        self.brush.setValue(0.3)
        self.brush.setSuffix(" m")
        self.brush.setToolTip("Radius of the brush, in metres.")
        size_row.addWidget(self.brush)
        size_row.addStretch(1)
        gl_.addLayout(size_row)
        hint = QLabel("Shift adds to the selection. Alt + drag rotates the view while a tool "
                      "is chosen; the right button pans and the wheel zooms as usual. Every "
                      "tool works in the walk-through too: there the right button looks "
                      "around.")
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        gl_.addWidget(hint)
        lay.addWidget(g)

        # ---- the selection
        g = QGroupBox("Selection")
        gl_ = QVBoxLayout(g)
        self.sel_label = QLabel("Nothing selected.")
        self.sel_label.setWordWrap(True)
        gl_.addWidget(self.sel_label)
        row = QHBoxLayout()
        self.b_delete = QPushButton("Delete")
        self.b_delete.setToolTip("Delete the selected points (Del). Every full-resolution "
                                 "point inside the selection is left out when you save.")
        self.b_delete.clicked.connect(self.delete_selection)
        self.b_restore = QPushButton("Restore")
        self.b_restore.setToolTip("Bring deleted points inside the selection back. Tick "
                                  "Show deleted points to see and select them.")
        self.b_restore.clicked.connect(self.restore_selection)
        self.b_clear = QPushButton("Clear")
        self.b_clear.setToolTip("Select nothing.")
        self.b_clear.clicked.connect(self.clear_selection)
        for b in (self.b_delete, self.b_restore, self.b_clear):
            row.addWidget(b)
        gl_.addLayout(row)
        self.cb_show_deleted = QCheckBox("Show deleted points (dark red)")
        self.cb_show_deleted.setToolTip("Draw the deleted points instead of hiding them, so "
                                        "they can be selected and restored.")
        self.cb_show_deleted.toggled.connect(self._show_deleted)
        gl_.addWidget(self.cb_show_deleted)
        self.scope_note = QLabel()
        self.scope_note.setWordWrap(True)
        self.scope_note.setStyleSheet(HINT_STYLE)
        gl_.addWidget(self.scope_note)
        lay.addWidget(g)

        # ---- history
        g = QGroupBox("Edits")
        gl_ = QVBoxLayout(g)
        self.history = QListWidget()
        self.history.setMinimumHeight(90)
        self.history.setMaximumHeight(170)
        self.history.setToolTip("Every delete and restore, oldest first. They are saved in "
                                "the project; the scan files are never changed.")
        gl_.addWidget(self.history)
        row = QHBoxLayout()
        self.b_undo = QPushButton("Undo")
        self.b_undo.setToolTip("Undo the last edit (Ctrl+Z).")
        self.b_undo.clicked.connect(self.undo)
        self.b_redo = QPushButton("Redo")
        self.b_redo.setToolTip("Redo what was undone (Ctrl+Y).")
        self.b_redo.clicked.connect(self.redo)
        row.addWidget(self.b_undo)
        row.addWidget(self.b_redo)
        row.addStretch(1)
        gl_.addLayout(row)
        lay.addWidget(g)

        # ---- split into parts
        g = QGroupBox("Split into parts")
        gl_ = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel("Split:"))
        self.split_target = QComboBox()
        self.split_target.setToolTip("What to cut in two: the whole site, or once it has "
                                     "parts, one of them.")
        row.addWidget(self.split_target, 1)
        gl_.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("At height:"))
        self.split_height = QDoubleSpinBox()
        self.split_height.setRange(-1e6, 1e6)
        self.split_height.setDecimals(2)
        self.split_height.setSingleStep(0.1)
        self.split_height.setSuffix(" m")
        self.split_height.setToolTip("A level cut at this height (for floors of a building). "
                                     "Colour by Height in the view helps to find it.")
        self.split_height.valueChanged.connect(lambda _v: setattr(self, "_height_set", True))
        self._height_set = False
        row.addWidget(self.split_height)
        self.b_split_h = QPushButton("Split here")
        self.b_split_h.clicked.connect(self.split_at_height)
        row.addWidget(self.b_split_h)
        gl_.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.tool_buttons["cut"])
        row.addStretch(1)
        gl_.addLayout(row)
        hint = QLabel("Parts are listed on the left: tick to show or hide them, double-click "
                      "to rename, right-click to save one on its own. Undo joins them again.")
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        gl_.addWidget(hint)
        lay.addWidget(g)

        # ---- automatic clean-up (its options live with the merge in AlignPanel)
        lay.addWidget(win.align_panel.cleanup_box)

        # ---- save
        self.b_save = QPushButton("Save cleaned cloud…")
        self.b_save.setToolTip("Save every scan, aligned, without the deleted points, as one "
                               "cloud (the same as Merge in Align, which also leaves them out).")
        self.b_save.clicked.connect(self._save)
        lay.addWidget(self.b_save)
        lay.addStretch(1)

        self.view.selection_changed.connect(self._selection_changed)
        self.refresh()

    # ---------------------------------------------------------------- mode

    def activate(self, on: bool) -> None:
        self._active = on
        self.view.set_edit_mode(on)
        if on:
            self.view.show_deleted = self.cb_show_deleted.isChecked()
            self._tool_toggled(self.tool_group.checkedId(), True)
            self._default_height()
        else:
            self.regions, self.kinds = [], []
        self.refresh()

    def _tool_toggled(self, k, checked):
        if not checked or not self._active:
            return
        key = TOOL_LABELS[k][0]
        self.view.set_tool(TOOLS[key](self.view, self) if key else None)

    def current_tool(self):
        """The chosen tool's key (None: Move the view)."""
        return TOOL_LABELS[max(self.tool_group.checkedId(), 0)][0]

    def choose_tool(self, key) -> None:
        self.tool_buttons[key].setChecked(True)

    def brush_radius(self) -> float:
        return float(self.brush.value())

    def _show_deleted(self, on):
        self.view.show_deleted = on
        self.view.set_selection(self.regions)

    # ---------------------------------------------------------------- selection

    def begin_selection(self, add: bool, kind: str) -> None:
        if not add and self.regions:
            self.regions, self.kinds = [], []
            self.view.set_selection([])
        if kind not in self.kinds:
            self.kinds.append(kind)

    def extend_selection(self, region) -> None:
        self.regions.append(region)
        self.view.extend_selection(region)

    def grow_at(self, seed, add: bool = False, points=None) -> None:
        """Select the object at ``seed``, grown over ``points`` (project
        frame; default: the points drawn in the orbit view)."""
        from PySide6.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            xyz = self.view.object_points(seed) if points is None else points
            region = grow_object(xyz, seed)
        finally:
            QApplication.restoreOverrideCursor()
        if not len(region.keys):
            self.win.tasks.set_status("No object found there.")
            return
        self.begin_selection(add, "object")
        self.extend_selection(region)

    def clear_selection(self) -> None:
        self.regions, self.kinds = [], []
        self.view.set_selection([])

    def _selection_changed(self):
        self.refresh()

    def selected_estimate(self) -> tuple[int, int]:
        """(points drawn and selected, full-resolution points that is about)."""
        shown = full = 0
        for i, (n, total) in self.view.selected_counts().items():
            if i < len(self.doc.scans) and total:
                shown += n
                full += n * self.doc.scans[i].points / total
        return shown, int(round(full))

    def _scope(self):
        """The scans an edit applies to: the visible ones (None: every scan)."""
        scans = self.doc.scans
        if all(s.entry.visible for s in scans):
            return None
        return [s.entry.key for s in scans if s.entry.visible]

    def _apply(self, action: str) -> None:
        if not self.regions:
            return
        _, full = self.selected_estimate()
        verb = "Deleted" if action == "delete" else "Restored"
        label = f"{verb} about {full:,} points ({', '.join(self.kinds) or 'selection'})"
        scope = self._scope()
        if scope is not None:
            label += f" in {len(scope)} scan(s)"
        self.doc.add_edit(Edit(action, union(self.regions), scans=scope, label=label))
        self.clear_selection()
        self.win.tasks.set_status(label + ".")

    def delete_selection(self) -> None:
        self._apply("delete")

    def restore_selection(self) -> None:
        self._apply("restore")

    def undo(self) -> None:
        e = self.doc.undo_edit()
        if e is not None:
            self.win.tasks.set_status("Undone: " + _describe(e))

    def redo(self) -> None:
        e = self.doc.redo_edit()
        if e is not None:
            self.win.tasks.set_status("Redone: " + _describe(e))

    def _save(self):
        self.win.align_panel.merge()

    # ---------------------------------------------------------------- splits

    def _split_within(self):
        """The part chosen to split (None: the whole site)."""
        pid = self.split_target.currentData()
        return self.doc.part(pid) if pid else None

    def split_with(self, plane: HalfSpace, names) -> None:
        within = self._split_within()
        if self.doc.edits.parts and within is None:
            return  # the site is already in parts: one of them must be chosen
        if within is not None:
            names = tuple(f"{within.name}: {n}" for n in names)
        a, b = self.doc.split(plane, names, within)
        self.win.tasks.set_status(f"Split into {a.name!r} and {b.name!r}. Untick one in the "
                                  "list on the left to hide it.")

    def split_at_height(self) -> None:
        h = float(self.split_height.value())
        self.split_with(HalfSpace([0, 0, 1], h), (f"above {h:.2f} m", f"below {h:.2f} m"))

    def _default_height(self) -> None:
        """Start the height half-way up the shown points (once per project)."""
        if self._height_set:
            return
        zs = [self.view.project_points(i)[0][:, 2] for i, s in enumerate(self.doc.scans)
              if s.entry.visible and len(s.sample)]
        if zs:
            z = np.concatenate(zs)
            self.split_height.blockSignals(True)
            self.split_height.setValue(float(np.median(z)))
            self.split_height.blockSignals(False)

    # ---------------------------------------------------------------- state

    def edits_changed(self):
        """The project's edits changed (or another project was opened)."""
        self.view.set_selection(self.regions)
        self.refresh()

    def refresh(self):
        doc = self.doc
        st = doc.edits
        shown, full = self.selected_estimate() if self.regions else (0, 0)
        if self.regions:
            self.sel_label.setText(f"{shown:,} points selected in the view, about "
                                   f"{full:,} at full resolution.")
        else:
            self.sel_label.setText("Nothing selected. Choose a tool above, then click or "
                                   "drag in the view.")
        scope = self._scope() if doc.scans else None
        self.scope_note.setText(
            "Edits apply to the visible scans only." if scope is not None else
            "Edits apply to every scan. Hide scans in the list to edit only the others.")
        self.history.clear()
        for e in st.point_edits:
            QListWidgetItem(_describe(e), self.history)
        if self.history.count():
            self.history.scrollToBottom()
        self.align_note.setVisible(len(doc.scans) > 1 and not doc.aligned)
        keep = self.split_target.currentData()
        self.split_target.blockSignals(True)
        self.split_target.clear()
        if not st.parts:
            self.split_target.addItem("The whole site", None)
        for p in st.parts:
            self.split_target.addItem(p.name, p.id)
        chosen = self.win.project_panel.selected_part() or keep
        k = self.split_target.findData(chosen) if chosen else -1
        self.split_target.setCurrentIndex(max(k, 0))
        self.split_target.blockSignals(False)
        self.update_buttons(self.win.tasks.busy if hasattr(self.win, "tasks") else False)

    def update_buttons(self, busy: bool):
        st = self.doc.edits
        has_sel = bool(self.regions)
        self.b_delete.setEnabled(has_sel and not busy)
        self.b_restore.setEnabled(has_sel and not busy)
        self.b_clear.setEnabled(has_sel)
        self.b_undo.setEnabled(st.can_undo() and not busy)
        self.b_redo.setEnabled(st.can_redo() and not busy)
        self.b_split_h.setEnabled(not busy and bool(self.doc.scans))
        self.tool_buttons["cut"].setEnabled(bool(self.doc.scans))
        self.b_save.setEnabled(not busy and bool(self.doc.scans)
                               and (self.doc.aligned or len(self.doc.scans) == 1))


def _describe(e) -> str:
    if isinstance(e, Edit):
        return e.label or ("Deleted points" if e.action == "delete" else "Restored points")
    return "Split into parts"
