"""Desktop app: drop PCD files in, align them, preview, then merge and save."""

from __future__ import annotations

import json
import os
import random
import sys
import traceback

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .merge import merge_clouds
from .pcd_io import PointCloud, read_pcd, write_pcd
from .registration import AlignResult, AlignSettings, align

try:
    import pyqtgraph.opengl as gl

    HAVE_GL = True
except Exception:  # pragma: no cover - depends on the machine's OpenGL
    HAVE_GL = False

PREVIEW_POINTS = 400_000  # total points drawn in the 3D view
PALETTE = [
    (0.95, 0.35, 0.25), (0.25, 0.65, 0.95), (0.35, 0.85, 0.35), (0.95, 0.80, 0.20),
    (0.75, 0.40, 0.90), (0.20, 0.85, 0.80), (0.95, 0.55, 0.75), (0.65, 0.65, 0.65),
]


# --------------------------------------------------------------------------
# Background work


class _Worker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            result = self.fn(*self.args, progress=self.progress.emit)
        except Exception as e:  # report every failure to the UI
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.finished.emit(result)


def _load_files(paths, progress):
    out, errors = [], []
    for k, p in enumerate(paths):
        progress(f"Loading {os.path.basename(p)}", k / max(1, len(paths)))
        try:
            out.append(read_pcd(p))
        except Exception as e:
            errors.append(f"{os.path.basename(p)}: {e}")
    return out, errors


def _align_clouds(clouds, settings, progress):
    xyz = [c.xyz[c.finite_mask()] for c in clouds]
    return align(xyz, settings, progress)


def _merge_and_save(clouds, transforms, path, add_source, compressed, progress):
    progress("Applying transforms to full-resolution clouds", 0.1)
    merged, partial = merge_clouds(clouds, transforms, add_source_field=add_source)
    progress(f"Writing {len(merged):,} points", 0.6)
    write_pcd(path, merged, compressed=compressed)
    return path, len(merged), partial


# --------------------------------------------------------------------------
# Main window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCD Merge")
        self.resize(1300, 820)
        self.setAcceptDrops(True)

        self.clouds: list[PointCloud] = []
        self.previews: list[np.ndarray] = []  # subsampled xyz per cloud
        self.result: AlignResult | None = None
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._items = []
        self._origin = np.zeros(3)

        # ---- left panel
        left = QWidget()
        lv = QVBoxLayout(left)

        self.drop_hint = QLabel("Drag and drop two or more .pcd files here\n(or use Add files…)")
        self.drop_hint.setAlignment(Qt.AlignCenter)
        self.drop_hint.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px; padding: 18px; color: #666; }"
        )
        lv.addWidget(self.drop_hint)

        self.file_list = QListWidget()
        self.file_list.itemChanged.connect(self._redraw)
        lv.addWidget(self.file_list, 1)

        row = QHBoxLayout()
        b_add = QPushButton("Add files…")
        b_add.clicked.connect(self._add_dialog)
        self.b_remove = QPushButton("Remove")
        self.b_remove.clicked.connect(self._remove_selected)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self._clear)
        for b in (b_add, self.b_remove, b_clear):
            row.addWidget(b)
        lv.addLayout(row)

        box = QGroupBox("Alignment")
        form = QFormLayout(box)
        self.voxel = QDoubleSpinBox()
        self.voxel.setDecimals(3)
        self.voxel.setRange(0.0, 100.0)
        self.voxel.setSingleStep(0.05)
        self.voxel.setSpecialValueText("Automatic")
        self.voxel.setToolTip(
            "Grid size used to thin the clouds for alignment only (same units as the "
            "scans, usually metres). Larger is faster and more robust, smaller is more precise. "
            "Merging always uses every original point."
        )
        form.addRow("Voxel size:", self.voxel)
        lv.addWidget(box)

        self.b_align = QPushButton("Align")
        self.b_align.setStyleSheet("font-weight: bold; padding: 8px;")
        self.b_align.clicked.connect(lambda: self._start_align(new_seed=False))
        lv.addWidget(self.b_align)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Alignment results will appear here.")
        self.report.setMaximumHeight(190)
        lv.addWidget(self.report)

        row = QHBoxLayout()
        self.b_retry = QPushButton("Retry")
        self.b_retry.setToolTip("Run the alignment again with a different random start.")
        self.b_retry.clicked.connect(lambda: self._start_align(new_seed=True))
        self.b_merge = QPushButton("Accept and merge…")
        self.b_merge.setStyleSheet("font-weight: bold; padding: 6px;")
        self.b_merge.clicked.connect(self._merge)
        row.addWidget(self.b_retry)
        row.addWidget(self.b_merge, 1)
        lv.addLayout(row)

        self.cb_source = QCheckBox("Add a 'source' field (which file each point came from)")
        self.cb_compress = QCheckBox("Write compressed PCD (binary_compressed)")
        lv.addWidget(self.cb_source)
        lv.addWidget(self.cb_compress)

        self.b_save_tf = QPushButton("Save transforms…")
        self.b_save_tf.clicked.connect(self._save_transforms)
        lv.addWidget(self.b_save_tf)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        lv.addWidget(self.progress)
        lv.addWidget(self.status)

        # ---- right panel: 3D view
        right = QWidget()
        rv = QVBoxLayout(right)
        top = QHBoxLayout()
        self.rb_orig = QRadioButton("Original positions")
        self.rb_aligned = QRadioButton("Aligned")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_orig)
        grp.addButton(self.rb_aligned)
        self.rb_orig.setChecked(True)
        self.rb_orig.toggled.connect(self._redraw)
        top.addWidget(QLabel("Show:"))
        top.addWidget(self.rb_orig)
        top.addWidget(self.rb_aligned)
        top.addStretch(1)
        top.addWidget(QLabel("Point size:"))
        self.pt_size = QSlider(Qt.Horizontal)
        self.pt_size.setRange(1, 8)
        self.pt_size.setValue(2)
        self.pt_size.setFixedWidth(120)
        self.pt_size.valueChanged.connect(self._redraw)
        top.addWidget(self.pt_size)
        b_fit = QPushButton("Reset view")
        b_fit.clicked.connect(self._fit_view)
        top.addWidget(b_fit)
        rv.addLayout(top)

        if HAVE_GL:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor((25, 25, 30))
            rv.addWidget(self.view, 1)
        else:
            self.view = None
            msg = QLabel("3D preview unavailable (OpenGL could not be initialised).")
            msg.setAlignment(Qt.AlignCenter)
            rv.addWidget(msg, 1)
        hint = QLabel("Left-drag: rotate   Right/middle-drag: pan   Wheel: zoom")
        hint.setStyleSheet("color: #888;")
        rv.addWidget(hint)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 920])
        self.setCentralWidget(split)

        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        self.addAction(quit_act)
        self._update_buttons()

    # ---------------------------------------------------------------- state

    def _busy(self) -> bool:
        return self._thread is not None

    def _update_buttons(self):
        busy = self._busy()
        n = len(self.clouds)
        self.b_align.setEnabled(not busy and n >= 2)
        self.b_retry.setEnabled(not busy and self.result is not None)
        self.b_merge.setEnabled(not busy and self.result is not None)
        self.b_save_tf.setEnabled(not busy and self.result is not None)
        self.b_remove.setEnabled(not busy and n > 0)
        self.rb_aligned.setEnabled(self.result is not None)
        self.drop_hint.setVisible(n == 0)

    def _set_status(self, msg: str, frac: float | None = None):
        self.status.setText(msg)
        if frac is not None:
            self.progress.setValue(int(frac * 100))

    def _run(self, fn, args, on_done, label):
        if self._busy():
            return
        self._thread = QThread()
        self._worker = _Worker(fn, *args)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._set_status)
        # Connect to bound methods (not lambdas) so Qt delivers the results
        # on the GUI thread.
        self._on_done = on_done
        self._worker.finished.connect(self._worker_done)
        self._worker.failed.connect(self._worker_failed)
        self._set_status(label, 0.0)
        self._thread.start()
        self._update_buttons()

    def _worker_done(self, payload):
        self._finish(self._on_done, payload)

    def _worker_failed(self, message):
        self._finish(None, message, failed=True)

    def _finish(self, on_done, payload, failed=False):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        self.progress.setValue(100 if not failed else 0)
        if failed:
            self._set_status("Failed: " + payload)
            QMessageBox.critical(self, "PCD Merge", payload)
        else:
            on_done(payload)
        self._update_buttons()

    # ---------------------------------------------------------------- files

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                paths += sorted(os.path.join(p, f) for f in os.listdir(p)
                                if f.lower().endswith(".pcd"))
            elif p.lower().endswith(".pcd"):
                paths.append(p)
        if paths:
            self.add_files(paths)
        else:
            self._set_status("Only .pcd files (or folders containing them) can be dropped.")

    def _add_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add point clouds", "",
                                                "Point clouds (*.pcd);;All files (*)")
        if paths:
            self.add_files(paths)

    def add_files(self, paths):
        known = {c.path for c in self.clouds}
        paths = [p for p in paths if p not in known]
        if paths and not self._busy():
            self._run(_load_files, (paths,), self._loaded, "Loading…")

    def _loaded(self, payload):
        clouds, errors = payload
        self.clouds.extend(clouds)
        self._rebuild_previews()
        self._invalidate()
        self._refresh_list()
        self._fit_view()
        if errors:
            QMessageBox.warning(self, "Some files could not be read", "\n".join(errors))
        self._set_status(f"{len(self.clouds)} cloud(s) loaded. "
                         + ("Press Align." if len(self.clouds) >= 2 else "Add at least one more."))

    def _rebuild_previews(self):
        if not self.clouds:
            return
        per = PREVIEW_POINTS // len(self.clouds)
        rng = np.random.default_rng(0)
        out = []
        for c in self.clouds:
            xyz = c.xyz[c.finite_mask()]
            if len(xyz) > per:
                xyz = xyz[rng.choice(len(xyz), per, replace=False)]
            out.append(xyz)
        self.previews = out
        # Draw relative to the reference cloud so large (e.g. UTM) coordinates
        # don't lose precision in float32 OpenGL buffers.
        self._origin = out[0].mean(axis=0) if len(out[0]) else np.zeros(3)

    def _refresh_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for i, c in enumerate(self.clouds):
            r, g, b = PALETTE[i % len(PALETTE)]
            pm = QPixmap(14, 14)
            pm.fill(QColor(int(r * 255), int(g * 255), int(b * 255)))
            text = f"{c.name}  ({len(c):,} pts)"
            if i == 0:
                text += "  [reference]"
            item = QListWidgetItem(QIcon(pm), text)
            item.setToolTip(f"{c.path}\nFields: {', '.join(c.field_names)}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        self._update_buttons()

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.file_list.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            del self.clouds[r]
        self._rebuild_previews()
        self._invalidate()
        self._refresh_list()
        self._redraw()

    def _clear(self):
        if self._busy():
            return
        self.clouds.clear()
        self.previews.clear()
        self._invalidate()
        self._refresh_list()
        self._redraw()
        self._set_status("Ready.", 0)

    def _invalidate(self):
        self.result = None
        self.report.clear()
        self.rb_orig.setChecked(True)

    # ---------------------------------------------------------------- align

    def _start_align(self, new_seed: bool):
        if len(self.clouds) < 2:
            return
        seed = random.randrange(1 << 30) if new_seed else 0
        settings = AlignSettings(voxel_size=self.voxel.value() or None, seed=seed)
        self._run(_align_clouds, (list(self.clouds), settings), self._aligned, "Aligning…")

    def _aligned(self, result: AlignResult):
        self.result = result
        lines = [f"Voxel size used: {result.voxel_size:.3g}", "", "Pair overlap:"]
        for p in result.pairs:
            lines.append(f"  {p.source + 1} ↔ {p.target + 1}: {p.fitness:.0%}"
                         f"  (rmse {p.rmse:.3g})")
        lines += ["", "Each cloud vs. the rest:"]
        for i, f in enumerate(result.cloud_fitness):
            lines.append(f"  {i + 1} {self.clouds[i].name}: {f:.0%}")
        warns = result.warnings()
        if warns:
            lines += ["", "Warnings:"] + ["  " + w for w in warns]
            lines += ["", "Try Retry, or a larger voxel size."]
        self.report.setPlainText("\n".join(lines))
        self.rb_aligned.setChecked(True)
        self._redraw()
        if warns:
            self._set_status("Aligned with warnings. Check the preview before merging.")
        else:
            self._set_status("Aligned. Check the preview, then Accept and merge (or Retry).")

    # ---------------------------------------------------------------- merge

    def _merge(self):
        if not self.result:
            return
        if not self.result.ok:
            ans = QMessageBox.question(
                self, "PCD Merge",
                "Some clouds could not be aligned and would be merged in their original "
                "position. Merge anyway?")
            if ans != QMessageBox.Yes:
                return
        base = os.path.dirname(self.clouds[0].path or "")
        path, _ = QFileDialog.getSaveFileName(self, "Save merged cloud",
                                              os.path.join(base, "merged.pcd"),
                                              "Point cloud (*.pcd)")
        if not path:
            return
        if not path.lower().endswith(".pcd"):
            path += ".pcd"
        compressed = self.cb_compress.isChecked()
        if compressed:
            try:
                import lzf  # noqa: F401
            except ImportError:
                QMessageBox.warning(self, "PCD Merge", "Compressed output needs python-lzf; "
                                    "writing uncompressed binary instead.")
                compressed = False
        self._run(_merge_and_save,
                  (list(self.clouds), list(self.result.transforms), path,
                   self.cb_source.isChecked(), compressed),
                  self._merged, "Merging…")

    def _merged(self, payload):
        path, n, partial = payload
        msg = f"Saved {n:,} points to {path}"
        if partial:
            msg += ("\n\nThese fields were missing from some files and were filled with 0: "
                    + ", ".join(partial))
        self._set_status(f"Saved {n:,} points to {os.path.basename(path)}.")
        QMessageBox.information(self, "PCD Merge", msg)

    def _save_transforms(self):
        if not self.result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save transforms", "transforms.json",
                                              "JSON (*.json)")
        if not path:
            return
        data = {
            "reference": self.clouds[0].path,
            "voxel_size": self.result.voxel_size,
            "transforms": {c.path: np.asarray(t).tolist()
                           for c, t in zip(self.clouds, self.result.transforms)},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._set_status(f"Transforms saved to {os.path.basename(path)}.")

    # ---------------------------------------------------------------- view

    def _redraw(self, *_):
        if self.view is None:
            return
        for it in self._items:
            self.view.removeItem(it)
        self._items = []
        aligned = self.rb_aligned.isChecked() and self.result is not None
        size = float(self.pt_size.value())
        for i, xyz in enumerate(self.previews):
            item = self.file_list.item(i)
            if item is not None and item.checkState() != Qt.Checked:
                continue
            pts = xyz
            if aligned:
                t = self.result.transforms[i]
                pts = xyz @ t[:3, :3].T + t[:3, 3]
            pts = (pts - self._origin).astype(np.float32)
            r, g, b = PALETTE[i % len(PALETTE)]
            sc = gl.GLScatterPlotItem(pos=pts, color=(r, g, b, 1.0), size=size, pxMode=True)
            sc.setGLOptions("opaque")
            self.view.addItem(sc)
            self._items.append(sc)

    def _fit_view(self):
        self._redraw()
        if self.view is None or not self.previews:
            return
        allp = np.concatenate(self.previews) - self._origin
        lo, hi = np.percentile(allp, 2, axis=0), np.percentile(allp, 98, axis=0)
        centre = (lo + hi) / 2
        from pyqtgraph import Vector

        self.view.setCameraPosition(pos=Vector(*centre), distance=float(np.linalg.norm(hi - lo)) * 1.1,
                                    elevation=35, azimuth=-60)


def run(argv=None) -> int:
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("PCD Merge")
    win = MainWindow()
    win.show()
    files = [a for a in (sys.argv[1:] if argv is None else argv[1:]) if a.lower().endswith(".pcd")]
    if files:
        win.add_files(files)
    return app.exec()
