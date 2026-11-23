"""The main window: menus, mode tabs, project tree on the left, the 3D view in
the middle, the current mode's settings on the right, and the task bar along
the bottom.

```
 File  View  Help                         [ Align ] [ Edit ] [ Inspect ]
 +-------------+--------------------------------+-----------------+
 | PROJECT     |  3D view (orbit / walk)        | SETTINGS for    |
 |  Scans      |                                | the mode        |
 |  Parts ...  |                                |                 |
 +-------------+--------------------------------+-----------------+
 status / progress / time left / Cancel
```

Each panel is its own module so features can be built side by side.
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QMainWindow, QMessageBox, QSplitter,
                               QStackedWidget, QTabBar)

from .. import __version__
from ..files import already_merged, read_manifest
from ..flyscene import build_scene
from ..project import SUFFIX, ProjectError, from_manifest, is_project, recent_projects
from ..readers import find_scan_files, input_filter
from .align_panel import AlignPanel
from .document import Document, load_clouds, load_files, sample_file, sample_missing
from .mode_panels import EditPanel, InspectPanel
from .project_panel import ProjectPanel
from .view3d import HAVE_GL, View3D
from .workers import TaskBar

MODES = ("Align", "Edit", "Inspect")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1440, 860)
        self.setAcceptDrops(True)
        self.doc = Document(self)

        self.tasks = TaskBar(self)
        self.statusBar().addWidget(self.tasks, 1)
        self.statusBar().setSizeGripEnabled(True)

        self.project_panel = ProjectPanel(self.doc)
        self.view = View3D(self.doc)
        self.align_panel = AlignPanel(self)
        self.edit_panel = EditPanel()
        self.inspect_panel = InspectPanel()
        self.right = QStackedWidget()
        for p in (self.align_panel, self.edit_panel, self.inspect_panel):
            self.right.addWidget(p)
        self.right.setMinimumWidth(max(340, self.align_panel.body.minimumSizeHint().width() + 24))

        split = QSplitter()
        split.addWidget(self.project_panel)
        split.addWidget(self.view)
        split.addWidget(self.right)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 760, 400])
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)

        self.modes = QTabBar()
        for m in MODES:
            self.modes.addTab(m)
        self.modes.setTabToolTip(0, "Get the scans into one frame, check the quality, merge "
                                    "and export.")
        self.modes.setTabToolTip(1, "Check and clean the scans: select, delete, split "
                                    "(selection tools come in 0.10).")
        self.modes.setTabToolTip(2, "Read-only, full-quality viewing and measuring (0.11).")
        self.modes.setExpanding(False)
        self.modes.setStyleSheet("QTabBar::tab { padding: 4px 18px; font-weight: bold; }")
        self.modes.currentChanged.connect(self.right.setCurrentIndex)
        self._build_menus()
        self.menuBar().setCornerWidget(self.modes, Qt.TopRightCorner)

        # wiring
        self.doc.changed.connect(self._doc_changed)
        self.doc.saved.connect(self._update_title)
        self.tasks.busy_changed.connect(self._update_buttons)
        self.project_panel.add_files.connect(self._add_dialog)
        self.project_panel.add_folder.connect(self._add_folder_dialog)
        self.project_panel.remove_scans.connect(self._remove_scans)
        self.project_panel.locate_scan.connect(self._locate)
        self.project_panel.open_export.connect(self._show_file)
        self.project_panel.export_toggled.connect(self.show_export)
        self.project_panel.remove_export.connect(self.doc.remove_export)
        self.align_panel.status.connect(self.tasks.set_status)
        self.view.walk_requested.connect(self.walk_through)

        self.align_panel.load_settings()
        self._update_title()
        self._update_buttons()

    # ---------------------------------------------------------------- menus

    def _action(self, menu, text, slot, shortcut=None, tip=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setStatusTip(tip)
            a.setToolTip(tip)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self._action(f, "New project", self.new_project, "Ctrl+N")
        self._action(f, "Open project…", self._open_dialog, "Ctrl+O")
        self.recent_menu = f.addMenu("Open recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        self.a_close = self._action(f, "Close project", self.close_project, "Ctrl+F4",
                                    "Close this project (it is saved first) and start an "
                                    "empty one.")
        self.a_save = self._action(f, "Save project", self.save_project, "Ctrl+S",
                                   "Projects save themselves after every change once they "
                                   "have a folder.")
        self._action(f, "Save project as…", self.save_project_as, "Ctrl+Shift+S")
        f.addSeparator()
        self.a_add = self._action(f, "Add scans…", self._add_dialog, "Ctrl+I")
        self.a_folder = self._action(f, "Add folder of scans…", self._add_folder_dialog)
        self._action(f, "Open a merged cloud as a new project…", self._import_merged,
                     tip="Start a project from a cloud merged earlier (and its .json scan "
                         "list), to add new scans to it.")
        f.addSeparator()
        self.a_copy = self._action(f, "Copy scans into the project", self._copy_scans_in,
                                   tip="Copy every scan file into the project folder, so the "
                                       "project can be handed over on its own.")
        self.a_clear_cache = self._action(f, "Clear the project's cache", self._clear_cache,
                                          tip="Delete the cached previews; they are rebuilt "
                                              "when needed.")
        f.addSeparator()
        self._action(f, "Quit", self.close, "Ctrl+Q")

        v = mb.addMenu("&View")
        self._action(v, "Reset view", self.view.fit, "Home")
        for k, name in enumerate(("Top", "Front", "Side"), start=1):
            self._action(v, f"{name} view", lambda _=False, i=k: (
                self.view.camera.setCurrentIndex(i), self.view._camera_preset()), f"Ctrl+{k}")
        self.a_walk = self._action(v, "Walk through", self.walk_through, "Ctrl+W")
        v.addSeparator()
        for k, m in enumerate(MODES):
            self._action(v, f"{m} mode", lambda _=False, i=k: self.modes.setCurrentIndex(i),
                         f"F{k + 2}")

        h = mb.addMenu("&Help")
        self._action(h, "User guide (online)", lambda: QDesktopServices.openUrl(
            QUrl("https://github.com/LeonSutliffe/PCD-Merge#readme")))
        self._action(h, "About PCD Merge", lambda: QMessageBox.about(
            self, "PCD Merge", f"PCD Merge {__version__}\n\nAlign, check and merge LiDAR scans."))

    def _fill_recent(self):
        self.recent_menu.clear()
        items = recent_projects()
        if not items:
            a = self.recent_menu.addAction("(none yet)")
            a.setEnabled(False)
        for p in items:
            a = self.recent_menu.addAction(os.path.basename(p))
            a.setToolTip(p)
            a.setStatusTip(p)
            a.triggered.connect(lambda _=False, path=p: self.open_project(path))

    # ---------------------------------------------------------------- state

    def _update_title(self):
        p = self.doc.project
        mark = " *" if self.doc.dirty and not p.saved else ""
        where = f" - {p.folder}" if p.saved else ""
        self.setWindowTitle(f"{p.name}{mark}{where} - PCD Merge {__version__}")
        self.project_panel.refresh()

    def _update_buttons(self, *_):
        busy = self.tasks.busy
        self.project_panel.set_busy(busy)
        self.align_panel.update_buttons(busy)
        self.view.update_controls(busy)
        for a in (self.a_add, self.a_folder):
            a.setEnabled(not busy)
        self.a_walk.setEnabled(not busy and bool(self.doc.scans) and HAVE_GL)
        self.a_copy.setEnabled(not busy and self.doc.project.saved and bool(self.doc.scans))
        self.a_clear_cache.setEnabled(self.doc.project.saved)
        self.a_close.setEnabled(not busy and (self.doc.project.saved or bool(self.doc.scans)))

    def _doc_changed(self, what):
        if what == "scans":
            self.view.leave_walk()
            self.align_panel.scans_changed()
            self.view.update_controls(self.tasks.busy)
            self.view.redraw()
        elif what in ("visibility", "exports"):
            self.view.redraw()
        elif what == "project":
            self.view.leave_walk()
            self.align_panel.load_settings()
            self.align_panel.cleanup_changed()
            self.view.update_controls(self.tasks.busy)
            if self.doc.aligned:
                self.view.show_aligned()
            self.view.fit()
        self._update_title()
        self._update_buttons()

    # ---------------------------------------------------------------- scans

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        dropped = [url.toLocalFile() for url in e.mimeData().urls()]
        projects = [p for p in dropped if is_project(p)]
        if projects:
            self.open_project(projects[0])
            return
        paths = find_scan_files(dropped)
        if paths:
            self.load_paths(paths)
        else:
            self.tasks.set_status("Only scans (PCD, PLY, LAS, LAZ, E57), folders of them or a "
                                  f"{SUFFIX} project can be dropped.")

    def _add_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add scans", "", input_filter())
        if paths:
            self.load_paths(paths)

    def _add_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Add every scan in a folder")
        if path:
            paths = find_scan_files([path])
            if paths:
                self.load_paths(paths)
            else:
                self.tasks.set_status("No scans found in that folder.")

    def load_paths(self, paths, base: bool = False):
        """Read new scans (or, with ``base``, an earlier merged cloud) into the project."""
        if self.tasks.busy:
            return
        doc = self.doc
        if not base:
            known = {os.path.normcase(os.path.abspath(s.entry.path)) for s in doc.scans
                     if s.entry.scans_in_file == 1}
            paths = [p for p in paths if os.path.normcase(os.path.abspath(p)) not in known]
            if doc.base is not None:
                skipped = already_merged(read_manifest(doc.base.entry.path), paths)
                paths = [p for p in paths if p not in skipped]
                if skipped:
                    self.tasks.set_status(f"Skipped {len(skipped)} file(s) already in the "
                                          "merged cloud.")
        if not paths:
            return
        self.tasks.run(load_files, (paths, base), lambda payload: self._loaded(payload, base),
                       "Loading merged cloud…" if base else "Loading…")

    def _loaded(self, payload, base):
        loaded, errors = payload
        doc = self.doc
        had = doc.keys()
        loaded = [(c, s) for c, s in loaded if base or _key(c) not in had]
        doc.add_loaded(loaded, base=base)
        if base and doc.base is not None:
            done = set(already_merged(read_manifest(doc.base.entry.path),
                                      [s.entry.path for s in doc.scans[1:]]))
            drop = [i for i, s in enumerate(doc.scans) if i and s.entry.path in done]
            if drop:
                doc.remove(drop)
        self.view.fit()
        if errors:
            QMessageBox.warning(self, "Some files could not be read", "\n".join(errors))
        n = len(doc.scans)
        if doc.base is not None and n < 2:
            self.tasks.set_status("Merged cloud loaded. Now add the new scans to append to it.")
        else:
            self.tasks.set_status(f"{n} scan(s) in the project. "
                                  + ("Press Align." if n >= 2 else "Add at least one more."))

    def _remove_scans(self, rows):
        if self.tasks.busy or not rows:
            return
        self.doc.remove(rows)

    def with_clouds(self, then, indices=None):
        """Run ``then()`` once the full clouds are loaded (after opening a
        project only previews are in memory)."""
        doc = self.doc
        need = doc.missing_clouds(indices)
        if not need:
            then()
            return
        missing = [doc.scans[i].name for i in need if doc.scans[i].entry.status() == "missing"]
        if missing:
            QMessageBox.warning(
                self, "PCD Merge", "These scans' files are missing: " + ", ".join(missing[:8])
                + ("…" if len(missing) > 8 else "")
                + ".\n\nDouble-click them in the project list to point to where they are now.")
            return

        def done(payload):
            clouds, errors = payload
            doc.set_clouds(clouds)
            if errors:
                QMessageBox.warning(self, "Some scans could not be read", "\n".join(errors))
                return
            then()

        self.tasks.run(load_clouds, ([(i, doc.scans[i].entry) for i in need],), done,
                       "Loading scans…")

    def _locate(self, i):
        e = self.doc.scans[i].entry
        ext = os.path.splitext(e.path)[1]
        path, _ = QFileDialog.getOpenFileName(
            self, f"Where is {os.path.basename(e.path)} now?", os.path.dirname(e.path),
            f"{os.path.basename(e.path)} (*{ext});;" + input_filter())
        if path:
            self.doc.relocate(i, path)
            self.doc.scans[i].cloud = None
            self.tasks.set_status(f"{os.path.basename(path)} found.")

    def _show_file(self, path):
        folder = os.path.dirname(path)
        if os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    # ---------------------------------------------------------------- walk-through

    def walk_through(self):
        doc = self.doc
        shown = [i for i, s in enumerate(doc.scans) if s.entry.visible]
        if not shown or self.tasks.busy:
            return
        aligned = self.view.aligned_shown()
        tfs = doc.transforms()
        self.with_clouds(lambda: self._walk_now(shown, aligned, tfs), shown)

    def _walk_now(self, shown, aligned, tfs):
        import numpy as np

        clouds = [self.doc.scans[i].cloud for i in shown]
        transforms = [tfs[i] if aligned else np.eye(4) for i in shown]
        title = ("Walk through: " + ", ".join(c.name for c in clouds[:3])
                 + (f" and {len(clouds) - 3} more" if len(clouds) > 3 else ""))
        self.tasks.run(build_scene, (clouds, transforms),
                       lambda scene: self._walk_ready(scene, title), "Preparing walk-through…")

    def _walk_ready(self, scene, title):
        self.view.show_walk(scene, title)
        self.tasks.set_status("Walk-through: click into the view, then use W A S D and the "
                              "mouse. Esc or Leave walk-through to go back.")

    # ---------------------------------------------------------------- projects

    def _confirm_discard(self) -> bool:
        """Before replacing an unsaved project that has work in it."""
        doc = self.doc
        if doc.project.saved or not doc.scans or not doc.dirty:
            if doc.project.saved and doc.dirty:
                doc.save()
            return True
        ans = QMessageBox.question(
            self, "PCD Merge", "Save this project first? Otherwise the list of scans and the "
            "alignment are lost (the scan files are never changed).",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if ans == QMessageBox.Save:
            return self.save_project_as()
        return ans == QMessageBox.Discard

    def new_project(self):
        if self.tasks.busy or not self._confirm_discard():
            return
        self.doc.new()
        self.tasks.set_status("New project. Add scans to start.")

    def close_project(self):
        if self.tasks.busy or not self._confirm_discard():
            return
        name = self.doc.project.name
        self.doc.new()
        self.tasks.set_status(f"{name} closed. Open a project or add scans to start a new one.")

    def show_export(self, path: str, on: bool = True):
        """Show or hide a merged cloud in the view (its preview is read once,
        streamed, then kept in the project's cache)."""
        self.doc.set_export_visible(path, on)
        if not on or self.doc.cached_export_sample(path) is not None:
            self.view.redraw()
            return
        if not os.path.isfile(path):
            self.tasks.set_status(f"{os.path.basename(path)} is no longer there.")
            return

        def done(payload):
            p, xyz = payload
            self.doc.set_export_sample(p, xyz)
            self.tasks.set_status(f"Showing {os.path.basename(p)}.")

        if not self.tasks.run(sample_file, (path,), done, "Reading the merged cloud…"):
            self.doc.set_export_visible(path, False)

    def _open_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, f"Open a project (a {SUFFIX} folder)", "",
            QFileDialog.ShowDirsOnly)
        if path:
            self.open_project(path)

    def open_project(self, path):
        if self.tasks.busy or not self._confirm_discard():
            return
        try:
            self.doc.open(path)
        except ProjectError as e:
            QMessageBox.warning(self, "PCD Merge", str(e))
            return
        problems = self.doc.project.check_scans()
        if problems:
            missing = [s.name for s, st in problems if st == "missing"]
            changed = [s.name for s, st in problems if st == "changed"]
            msg = []
            if missing:
                msg.append("These scans' files are missing (moved or deleted): "
                           + ", ".join(missing[:10]) + ("…" if len(missing) > 10 else "")
                           + ". Double-click them in the project list to point to them.")
            if changed:
                msg.append("These files changed since they were added, so the saved "
                           "alignment may not fit them: " + ", ".join(changed[:10]))
            QMessageBox.warning(self, "PCD Merge", "\n\n".join(msg))
        self.tasks.set_status(f"Opened {self.doc.project.name}.")
        need = [(i, s.entry) for i, s in enumerate(self.doc.scans)
                if not len(s.sample) and s.entry.status() != "missing"]
        if need:
            self.tasks.run(sample_missing, (need,), self._sampled, "Reading previews…")

    def _sampled(self, samples):
        self.doc.set_samples(samples)
        self.view.fit()
        self.tasks.set_status(f"Opened {self.doc.project.name}.")

    def save_project(self) -> bool:
        if self.doc.project.saved:
            ok = self.doc.save()
            self.tasks.set_status("Project saved." if ok else "The project could not be saved.")
            return ok
        return self.save_project_as()

    def save_project_as(self) -> bool:
        start = ""
        if self.doc.scans:
            start = os.path.join(os.path.dirname(self.doc.scans[0].entry.path), "Site" + SUFFIX)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project as", start, f"PCD Merge project (*{SUFFIX})")
        if not path:
            return False
        try:
            self.doc.save_as(path)
        except (ProjectError, OSError) as e:
            QMessageBox.warning(self, "PCD Merge", str(e))
            return False
        self.tasks.set_status(f"Project saved to {self.doc.project.folder}. From now on it "
                              "saves itself after every change.")
        return True

    def _import_merged(self):
        if self.tasks.busy or not self._confirm_discard():
            return
        merged, _ = QFileDialog.getOpenFileName(self, "Merged cloud", "", input_filter())
        if not merged:
            return
        start = os.path.splitext(merged)[0] + SUFFIX
        path, _ = QFileDialog.getSaveFileName(self, "New project for it", start,
                                              f"PCD Merge project (*{SUFFIX})")
        if not path:
            return
        try:
            project = from_manifest(merged, path)
        except (ProjectError, OSError, ValueError) as e:
            QMessageBox.warning(self, "PCD Merge", str(e))
            return
        self.open_project(project.folder)
        self.tasks.set_status("Merged cloud opened as a project. Add the new scans and press "
                              "Align; the merged cloud stays where it is.")

    def _copy_scans_in(self):
        doc = self.doc

        def copy(progress):
            doc.project.copy_scans_in(progress)

        def done(_):
            doc.touch("scans")
            self.tasks.set_status("Scans copied into the project folder.")

        self.tasks.run(copy, (), done, "Copying scans…")

    def _clear_cache(self):
        self.doc.project.clear_cache()
        self.tasks.set_status("Cache cleared. Previews are rebuilt the next time the project "
                              "opens.")

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        self.view.leave_walk()
        self.align_panel.shutdown()
        self.tasks.wait()
        super().closeEvent(e)


def _key(cloud) -> str:
    p = os.path.normcase(os.path.abspath(cloud.path))
    return f"{p}#{cloud.extra['scan_index']}" if cloud.extra.get("scans_in_file", 1) > 1 else p


def run(argv=None) -> int:
    args = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(args)
    app.setApplicationName("PCD Merge")
    from ..startup import before_main_window

    if before_main_window(app):
        return 0  # installing or updating; the new copy starts by itself
    win = MainWindow()
    win.show()
    rest = args[1:]
    projects = [a for a in rest if is_project(a)]
    if projects:
        win.open_project(projects[0])
    else:
        files = find_scan_files([a for a in rest if os.path.isfile(a)])
        if files:
            win.load_paths(files)
    return app.exec()
