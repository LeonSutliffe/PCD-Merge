"""Left side of the window: the project tree, the same in every mode.

Scans (tick to show or hide), parts (splits made in Edit), measurements
(0.11) and merged clouds.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QMenu, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .common import export_icon, swatch_icon

ROLE_INDEX = Qt.UserRole + 1


class ProjectPanel(QWidget):
    add_files = Signal()
    add_folder = Signal()
    remove_scans = Signal(list)  # scan indices
    locate_scan = Signal(int)
    open_export = Signal(str)
    export_toggled = Signal(str, bool)
    remove_export = Signal(str)
    save_part = Signal(str)  # part id

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 0, 4)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: bold;")
        lay.addWidget(self.title)

        self.drop_hint = QLabel("Drag and drop two or more scans\n(PCD, PLY, LAS, LAZ, E57)\n"
                                "or a whole folder here,\nor use Add files…")
        self.drop_hint.setAlignment(Qt.AlignCenter)
        self.drop_hint.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px; padding: 14px; color: #666; }")
        lay.addWidget(self.drop_hint)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        lay.addWidget(self.tree, 1)

        self.g_scans = QTreeWidgetItem(["Scans"])
        self.g_parts = QTreeWidgetItem(["Parts"])
        self.g_meas = QTreeWidgetItem(["Measurements"])
        self.g_exports = QTreeWidgetItem(["Merged clouds"])
        for g in (self.g_scans, self.g_parts, self.g_meas, self.g_exports):
            g.setFlags(Qt.ItemIsEnabled)
            self.tree.addTopLevelItem(g)
        self.g_parts.setToolTip(0, "Pieces of the site (floors, wings) made by splitting in "
                                   "Edit mode. Tick to show; double-click to rename; "
                                   "right-click to save one on its own.")
        self.g_meas.setToolTip(0, "Measurements made in Inspect mode (coming in 0.11).")
        self.g_exports.setToolTip(0, "Clouds merged or exported from this project. Tick one "
                                     "to show it in the 3D view.")

        row = QHBoxLayout()
        self.b_add = QPushButton("Add files…")
        self.b_add.clicked.connect(self.add_files)
        self.b_folder = QPushButton("Add folder…")
        self.b_folder.setToolTip("Add every scan (PCD, PLY, LAS, LAZ, E57) in a folder and its "
                                 "subfolders.")
        self.b_folder.clicked.connect(self.add_folder)
        self.b_remove = QPushButton("Remove")
        self.b_remove.setToolTip("Take the selected scans out of the project (the files are "
                                 "not touched).")
        self.b_remove.clicked.connect(self._remove_selected)
        for b in (self.b_add, self.b_folder, self.b_remove):
            row.addWidget(b)
        lay.addLayout(row)
        self.refresh()

    # ---------------------------------------------------------------- refresh

    def refresh(self):
        doc = self.doc
        self.title.setText(f"Project: {doc.project.name}" + (" *" if doc.dirty and
                                                               not doc.project.saved else ""))
        self.tree.blockSignals(True)
        expanded = self.g_scans.isExpanded() or not self.g_scans.childCount()
        self.g_scans.takeChildren()
        missing = {s.key for s, st in doc.project.check_scans() if st == "missing"} \
            if doc.project.saved else set()
        for i, scan in enumerate(doc.scans):
            e = scan.entry
            text = f"{scan.name}  ({scan.points:,} pts)"
            if e.base:
                text += "  [existing merged cloud, stays fixed]"
            elif i == 0:
                text += "  [reference]"
            if e.key in missing:
                text += "  [file missing: double-click to locate]"
            item = QTreeWidgetItem([text])
            item.setIcon(0, swatch_icon(i))
            item.setData(0, ROLE_INDEX, i)
            item.setToolTip(0, f"{e.path}\nFields: {', '.join(e.fields)}")
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if e.visible else Qt.Unchecked)
            if e.key in missing:
                item.setForeground(0, Qt.red)
            self.g_scans.addChild(item)
        self.g_scans.setText(0, f"Scans ({len(doc.scans)})")
        self.g_exports.takeChildren()
        for ex in reversed(doc.project.exports[-20:]):
            gone = not os.path.isfile(ex["path"])
            text = f"{os.path.basename(ex['path'])}  ({ex['points']:,} pts)"
            it = QTreeWidgetItem([text + ("  [file missing]" if gone else "")])
            it.setIcon(0, export_icon())
            it.setToolTip(0, f"{ex['path']}\nSaved {ex.get('time', '').replace('T', ' ')}\n"
                             "Tick to show it in the 3D view. Double-click to open its folder.")
            it.setData(0, ROLE_INDEX, ex["path"])
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            if not gone:
                flags |= Qt.ItemIsUserCheckable
                it.setCheckState(0, Qt.Checked if ex.get("visible") else Qt.Unchecked)
            it.setFlags(flags)
            self.g_exports.addChild(it)
        self.g_exports.setText(0, f"Merged clouds ({len(doc.project.exports)})")
        self.g_parts.takeChildren()
        parts = doc.project.edits.parts
        for p in parts:
            it = QTreeWidgetItem([p.name])
            it.setData(0, ROLE_INDEX, p.id)
            it.setToolTip(0, "Tick to show this part. Double-click to rename it; right-click "
                             "to save it on its own or split it again.")
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                        | Qt.ItemIsEditable)
            it.setCheckState(0, Qt.Checked if p.visible else Qt.Unchecked)
            self.g_parts.addChild(it)
        self.g_parts.setText(0, f"Parts ({len(parts)})" if parts else "Parts")
        self.g_parts.setDisabled(not parts)
        self.g_parts.setExpanded(True)
        self.g_meas.setDisabled(True)
        self.g_scans.setExpanded(expanded)
        self.g_exports.setExpanded(True)
        self.tree.blockSignals(False)
        self.drop_hint.setVisible(not doc.scans)

    def set_busy(self, busy: bool):
        for b in (self.b_add, self.b_folder, self.b_remove):
            b.setEnabled(not busy)
        self.b_remove.setEnabled(not busy and bool(self.doc.scans))

    # ---------------------------------------------------------------- events

    def _scan_index(self, item) -> int | None:
        if item is None or item.parent() is not self.g_scans:
            return None
        return item.data(0, ROLE_INDEX)

    def selected_scans(self) -> list[int]:
        return sorted(i for it in self.tree.selectedItems()
                      if (i := self._scan_index(it)) is not None)

    def _item_changed(self, item, _col):
        i = self._scan_index(item)
        if i is not None:
            self.doc.set_visible(i, item.checkState(0) == Qt.Checked)
        elif item.parent() is self.g_exports:
            self.export_toggled.emit(item.data(0, ROLE_INDEX), item.checkState(0) == Qt.Checked)
        elif item.parent() is self.g_parts:
            pid = item.data(0, ROLE_INDEX)
            part = self.doc.part(pid)
            if part is None:
                return
            if item.text(0).strip() and item.text(0) != part.name:
                self.doc.rename_part(pid, item.text(0))
            self.doc.set_part_visible(pid, item.checkState(0) == Qt.Checked)

    def selected_part(self) -> str | None:
        for it in self.tree.selectedItems():
            if it.parent() is self.g_parts:
                return it.data(0, ROLE_INDEX)
        return None

    def _double_clicked(self, item, _col):
        i = self._scan_index(item)
        if i is not None and self.doc.scans[i].entry.status() == "missing":
            self.locate_scan.emit(i)
        elif item.parent() is self.g_exports:
            self.open_export.emit(item.data(0, ROLE_INDEX))

    def _remove_selected(self):
        rows = self.selected_scans()
        if rows:
            self.remove_scans.emit(rows)

    def _menu(self, pos):
        item = self.tree.itemAt(pos)
        i = self._scan_index(item)
        if item is not None and item.parent() is self.g_parts:
            pid = item.data(0, ROLE_INDEX)
            m = QMenu(self)
            a = QAction("Save this part…", m)
            a.triggered.connect(lambda: self.save_part.emit(pid))
            m.addAction(a)
            a = QAction("Rename", m)
            a.triggered.connect(lambda: self.tree.editItem(item, 0))
            m.addAction(a)
            m.exec(self.tree.viewport().mapToGlobal(pos))
            return
        if item is not None and item.parent() is self.g_exports:
            path = item.data(0, ROLE_INDEX)
            m = QMenu(self)
            a = QAction("Open its folder", m)
            a.triggered.connect(lambda: self.open_export.emit(path))
            m.addAction(a)
            a = QAction("Remove from list (the file is kept)", m)
            a.triggered.connect(lambda: self.remove_export.emit(path))
            m.addAction(a)
            m.exec(self.tree.viewport().mapToGlobal(pos))
            return
        if i is None:
            return
        m = QMenu(self)
        a = QAction("Locate file…", m)
        a.triggered.connect(lambda: self.locate_scan.emit(i))
        m.addAction(a)
        a = QAction("Remove from project", m)
        a.triggered.connect(lambda: self.remove_scans.emit(self.selected_scans() or [i]))
        m.addAction(a)
        m.exec(self.tree.viewport().mapToGlobal(pos))
